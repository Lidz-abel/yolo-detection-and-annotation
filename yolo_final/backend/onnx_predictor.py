"""Reserved ONNX predictor slot for checkpoint 8 export artifacts."""

from __future__ import annotations

from PIL import Image

from backend.predictor_base import BasePredictor


class ONNXPredictor(BasePredictor):
    """Placeholder that preserves the future ONNX integration boundary."""

    def __init__(self, model_path, *_, **__):
        self.model_path = model_path

    def predict(
        self,
        image: Image.Image,
        score_threshold: float,
        top_k: int,
        nms_iou_threshold: float,
    ) -> dict:
        raise NotImplementedError(
            "ONNX inference is reserved for checkpoint 8. "
            "Set YOLO_BACKEND_MODEL_FORMAT=pytorch for the current backend."
        )

