"""Current PyTorch checkpoint predictor used before ONNX export is available."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from backend.predictor_base import BasePredictor
from models.detector import YOLOv0Baseline
from utils.config import load_config, parse_anchor_map, parse_anchor_string, parse_int_list, parse_string_list
from utils.prediction import decode_predictions_for_image, select_prediction_for_image


class PyTorchPredictor(BasePredictor):
    """Load a training config plus `.pth` checkpoint and run one-image inference."""

    def __init__(
        self,
        config_path: Path,
        checkpoint_path: Path,
        device_name: str,
        metadata_path: Path | None = None,
        use_fp16: bool = True,
    ):
        self.config_path = Path(config_path).resolve()
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.metadata_path = Path(metadata_path).resolve() if metadata_path else None
        self.config = load_config(self.config_path)
        self.data_cfg = self.config["data"]
        self.model_cfg = self.config["model"]
        self.evaluation_cfg = self.config["evaluation"]
        self.image_size = int(self.data_cfg["image_size"])
        self.num_classes = int(self.data_cfg["num_classes"])
        self.num_boxes = int(self.model_cfg.get("num_boxes", 1))
        self.feature_levels = self._resolve_feature_levels()
        self.anchors = parse_anchor_string(self.model_cfg.get("anchors"))
        self.anchors_by_level = parse_anchor_map(self.model_cfg, self.feature_levels)
        self.box_parameterization = str(self.model_cfg.get("box_parameterization", "legacy"))
        self.device = self._build_device(device_name)
        self.use_fp16 = bool(use_fp16) and self.device.type == "cuda"
        self.class_names = self._load_class_names()
        self.model = self._load_model()
        self._lock = threading.Lock()

    def _resolve_feature_levels(self) -> list[str]:
        grid_sizes = parse_int_list(self.data_cfg.get("grid_sizes"))
        feature_levels = parse_string_list(self.model_cfg.get("feature_levels"))
        if feature_levels:
            return feature_levels
        if grid_sizes:
            return [f"scale_{index}" for index in range(len(grid_sizes))]
        return ["scale_0"]

    @staticmethod
    def _build_device(device_name: str) -> torch.device:
        if str(device_name).lower() == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device_name)

    def _load_model(self):
        if not self.config_path.exists():
            raise FileNotFoundError(f"config not found: {self.config_path}")
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"checkpoint not found: {self.checkpoint_path}")
        model = YOLOv0Baseline(
            num_classes=self.num_classes,
            model_name=str(self.model_cfg["name"]),
            width_mult=float(self.model_cfg["width_mult"]),
            depth_mult=float(self.model_cfg["depth_mult"]),
            use_residual=bool(self.model_cfg["use_residual"]),
            num_boxes=self.num_boxes,
            head_type=str(self.model_cfg.get("head_type", "shared")),
            neck_type=str(self.model_cfg.get("neck_type", "none")),
            feature_levels=self.feature_levels,
        ).to(self.device)
        state_dict = torch.load(self.checkpoint_path, map_location=self.device)
        state_dict = {
            key.removeprefix("module."): value
            for key, value in state_dict.items()
        }
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def _load_class_names(self) -> dict[int, str]:
        if self.metadata_path is None or not self.metadata_path.exists():
            return {}
        try:
            payload = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            coco_info = payload["coco2017"]
            mapping = coco_info["contiguous_category_id_to_name_by_split"].get("val", {})
            return {int(class_id): str(name) for class_id, name in mapping.items()}
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return {}

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        resized = image.convert("RGB").resize((self.image_size, self.image_size), Image.BILINEAR)
        array = np.asarray(resized, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)

    def _decode(self, pred, score_threshold: float, top_k: int, nms_iou_threshold: float) -> list[dict]:
        return decode_predictions_for_image(
            pred=pred,
            image_size=self.image_size,
            num_classes=self.num_classes,
            num_boxes=self.num_boxes,
            anchors=self.anchors_by_level if self.anchors_by_level else self.anchors,
            box_parameterization=self.box_parameterization,
            score_threshold=score_threshold,
            top_k=top_k,
            nms_iou_threshold=nms_iou_threshold,
            score_alpha=float(self.evaluation_cfg.get("score_alpha", 1.0)),
            score_beta=float(self.evaluation_cfg.get("score_beta", 1.0)),
        )

    def predict(
        self,
        image: Image.Image,
        score_threshold: float,
        top_k: int,
        nms_iou_threshold: float,
    ) -> dict:
        total_start = time.perf_counter()
        original_width, original_height = image.size
        preprocess_start = time.perf_counter()
        image_tensor = self._preprocess(image)
        preprocess_ms = (time.perf_counter() - preprocess_start) * 1000.0

        with self._lock, torch.inference_mode():
            inference_start = time.perf_counter()
            autocast_enabled = self.use_fp16
            with torch.autocast(device_type=self.device.type, enabled=autocast_enabled):
                raw_pred = self.model(image_tensor)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            inference_ms = (time.perf_counter() - inference_start) * 1000.0
            postprocess_start = time.perf_counter()
            pred = select_prediction_for_image(raw_pred, 0)
            decoded = self._decode(pred, score_threshold, top_k, nms_iou_threshold)

        scale_x = float(original_width) / float(self.image_size)
        scale_y = float(original_height) / float(self.image_size)
        bboxes = []
        for item in decoded:
            x1, y1, x2, y2 = item["box_xyxy"]
            x1 = max(0.0, min(float(x1) * scale_x, float(original_width)))
            x2 = max(0.0, min(float(x2) * scale_x, float(original_width)))
            y1 = max(0.0, min(float(y1) * scale_y, float(original_height)))
            y2 = max(0.0, min(float(y2) * scale_y, float(original_height)))
            if x2 <= x1 or y2 <= y1:
                continue
            class_id = int(item["class_id"])
            bboxes.append(
                {
                    "class_id": class_id,
                    "class_name": self.class_names.get(class_id, str(class_id)),
                    "score": float(item["score"]),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                }
            )

        postprocess_ms = (time.perf_counter() - postprocess_start) * 1000.0
        total_ms = (time.perf_counter() - total_start) * 1000.0

        return {
            "image_width": original_width,
            "image_height": original_height,
            "bboxes": bboxes,
            "latency_ms": {
                "preprocess": preprocess_ms,
                "inference": inference_ms,
                "postprocess": postprocess_ms,
                "total": total_ms,
            },
            "model": {
                "format": "pytorch",
                "config": str(self.config_path),
                "checkpoint": str(self.checkpoint_path),
                "device": str(self.device),
                "fp16": self.use_fp16,
            },
        }
