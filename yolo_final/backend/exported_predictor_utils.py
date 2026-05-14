"""Shared helpers for exported-model inference backends."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from utils.config import load_config, parse_anchor_map, parse_anchor_string, parse_string_list
from utils.prediction import decode_predictions_for_image, select_prediction_for_image


class DetectionRuntimeMixin:
    """Common preprocessing, decoding, and response formatting for predictors."""

    def _init_runtime_common(self, config_path: Path, metadata_path: Path | None = None):
        self.config_path = Path(config_path).resolve()
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
        self.class_names = self._load_class_names()
        self._lock = threading.Lock()

    def _resolve_feature_levels(self) -> list[str]:
        feature_levels = parse_string_list(self.model_cfg.get("feature_levels"))
        if feature_levels:
            return feature_levels
        grid_sizes = self.data_cfg.get("grid_sizes")
        if grid_sizes:
            return [f"scale_{index}" for index, _ in enumerate(str(grid_sizes).split(","))]
        return ["scale_0"]

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

    def _preprocess_numpy(self, image: Image.Image) -> np.ndarray:
        resized = image.convert("RGB").resize((self.image_size, self.image_size), Image.BILINEAR)
        array = np.asarray(resized, dtype=np.float32) / 255.0
        return np.transpose(array, (2, 0, 1))[None, ...]

    def _preprocess_torch(self, image: Image.Image, device: torch.device) -> torch.Tensor:
        return torch.from_numpy(self._preprocess_numpy(image)).to(device)

    def _outputs_to_prediction(self, raw_outputs):
        if isinstance(raw_outputs, dict):
            return select_prediction_for_image(raw_outputs, 0)
        if isinstance(raw_outputs, (tuple, list)):
            return {
                level: raw_outputs[index][0].detach().cpu()
                for index, level in enumerate(self.feature_levels)
            }
        return raw_outputs[0].detach().cpu()

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

    def _format_response(
        self,
        image: Image.Image,
        decoded: list[dict],
        latency_ms: dict[str, float],
        model_info: dict,
    ) -> dict:
        original_width, original_height = image.size
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
        return {
            "image_width": original_width,
            "image_height": original_height,
            "bboxes": bboxes,
            "latency_ms": latency_ms,
            "model": model_info,
        }

    @staticmethod
    def _elapsed_ms(start: float) -> float:
        return (time.perf_counter() - start) * 1000.0
