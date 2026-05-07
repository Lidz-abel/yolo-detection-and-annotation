"""Predictor abstraction used by the Flask API."""

from __future__ import annotations

from abc import ABC, abstractmethod
from PIL import Image


class BasePredictor(ABC):
    """Common interface for PyTorch, TorchScript, and ONNX predictors."""

    @abstractmethod
    def predict(
        self,
        image: Image.Image,
        score_threshold: float,
        top_k: int,
        nms_iou_threshold: float,
    ) -> dict:
        """Return model predictions for one PIL image."""

