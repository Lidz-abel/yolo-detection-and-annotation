"""Current PyTorch checkpoint predictor used before ONNX export is available."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from PIL import Image

from backend.exported_predictor_utils import DetectionRuntimeMixin
from backend.predictor_base import BasePredictor
from models.detector import YOLOv0Baseline
from utils.config import load_config, parse_int_list, parse_string_list
from utils.prediction import select_prediction_for_image


class PyTorchPredictor(DetectionRuntimeMixin, BasePredictor):
    """Load a training config plus `.pth` checkpoint and run one-image inference."""

    def __init__(
        self,
        config_path: Path,
        checkpoint_path: Path,
        device_name: str,
        metadata_path: Path | None = None,
        use_fp16: bool = True,
    ):
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self._init_runtime_common(Path(config_path), metadata_path)
        self.device = self._build_device(device_name)
        self.use_fp16 = bool(use_fp16) and self.device.type == "cuda"
        self.model = self._load_model()

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

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        return self._preprocess_torch(image, self.device)

    def predict(
        self,
        image: Image.Image,
        score_threshold: float,
        top_k: int,
        nms_iou_threshold: float,
    ) -> dict:
        total_start = time.perf_counter()
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

        postprocess_ms = (time.perf_counter() - postprocess_start) * 1000.0
        total_ms = (time.perf_counter() - total_start) * 1000.0

        return self._format_response(
            image=image,
            decoded=decoded,
            latency_ms={
                "preprocess": preprocess_ms,
                "inference": inference_ms,
                "postprocess": postprocess_ms,
                "total": total_ms,
            },
            model_info={
                "format": "pytorch",
                "config": str(self.config_path),
                "checkpoint": str(self.checkpoint_path),
                "device": str(self.device),
                "fp16": self.use_fp16,
            },
        )
