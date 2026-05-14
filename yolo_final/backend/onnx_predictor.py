"""ONNX Runtime predictor for checkpoint 8 export artifacts."""

from __future__ import annotations

import time
from pathlib import Path

import torch
from PIL import Image

from backend.exported_predictor_utils import DetectionRuntimeMixin
from backend.predictor_base import BasePredictor


class ONNXPredictor(DetectionRuntimeMixin, BasePredictor):
    """Load an ONNX detector and reuse Python post-processing."""

    def __init__(
        self,
        model_path: Path,
        config_path: Path,
        metadata_path: Path | None = None,
        providers: list[str] | None = None,
    ):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "onnxruntime is required for YOLO_BACKEND_MODEL_FORMAT=onnx. "
                "Install onnxruntime-gpu or onnxruntime in the yolov1 environment."
            ) from exc
        self.model_path = Path(model_path).resolve()
        self._init_runtime_common(Path(config_path), metadata_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")
        available = ort.get_available_providers()
        if providers is None:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        providers = [provider for provider in providers if provider in available]
        if not providers:
            providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(self.model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        self.providers = self.session.get_providers()

    def predict(
        self,
        image: Image.Image,
        score_threshold: float,
        top_k: int,
        nms_iou_threshold: float,
    ) -> dict:
        total_start = time.perf_counter()
        preprocess_start = time.perf_counter()
        image_array = self._preprocess_numpy(image)
        preprocess_ms = self._elapsed_ms(preprocess_start)

        with self._lock:
            inference_start = time.perf_counter()
            outputs = self.session.run(self.output_names, {self.input_name: image_array})
            inference_ms = self._elapsed_ms(inference_start)
            postprocess_start = time.perf_counter()
            tensor_outputs = tuple(torch.from_numpy(output) for output in outputs)
            decoded = self._decode(
                self._outputs_to_prediction(tensor_outputs),
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
                "format": "onnx",
                "model_path": str(self.model_path),
                "config": str(self.config_path),
                "providers": self.providers,
            },
        )
