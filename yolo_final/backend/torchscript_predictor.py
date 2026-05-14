"""TorchScript exported-model predictor for the Flask backend."""

from __future__ import annotations

import time
from pathlib import Path

import torch
from PIL import Image

from backend.exported_predictor_utils import DetectionRuntimeMixin
from backend.predictor_base import BasePredictor


class TorchScriptPredictor(DetectionRuntimeMixin, BasePredictor):
    """Load a traced TorchScript detector and reuse Python post-processing."""

    def __init__(
        self,
        model_path: Path,
        config_path: Path,
        device_name: str,
        metadata_path: Path | None = None,
        use_fp16: bool = True,
    ):
        self.model_path = Path(model_path).resolve()
        self._init_runtime_common(Path(config_path), metadata_path)
        self.device = self._build_device(device_name)
        self.use_fp16 = bool(use_fp16) and self.device.type == "cuda"
        if not self.model_path.exists():
            raise FileNotFoundError(f"TorchScript model not found: {self.model_path}")
        self.model = torch.jit.load(str(self.model_path), map_location=self.device)
        self.model.eval()

    @staticmethod
    def _build_device(device_name: str) -> torch.device:
        if str(device_name).lower() == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device_name)

    def predict(
        self,
        image: Image.Image,
        score_threshold: float,
        top_k: int,
        nms_iou_threshold: float,
    ) -> dict:
        total_start = time.perf_counter()
        preprocess_start = time.perf_counter()
        image_tensor = self._preprocess_torch(image, self.device)
        preprocess_ms = self._elapsed_ms(preprocess_start)

        with self._lock, torch.inference_mode():
            inference_start = time.perf_counter()
            with torch.autocast(device_type=self.device.type, enabled=self.use_fp16):
                raw_outputs = self.model(image_tensor)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            inference_ms = self._elapsed_ms(inference_start)
            postprocess_start = time.perf_counter()
            decoded = self._decode(
                self._outputs_to_prediction(raw_outputs),
                score_threshold,
                top_k,
                nms_iou_threshold,
            )
            postprocess_ms = self._elapsed_ms(postprocess_start)

        return self._format_response(
            image=image,
            decoded=decoded,
            latency_ms={
                "preprocess": preprocess_ms,
                "inference": inference_ms,
                "postprocess": postprocess_ms,
                "total": self._elapsed_ms(total_start),
            },
            model_info={
                "format": "torchscript",
                "model_path": str(self.model_path),
                "config": str(self.config_path),
                "device": str(self.device),
                "fp16": self.use_fp16,
            },
        )
