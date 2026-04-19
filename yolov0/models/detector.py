"""Compose a selected backbone and the detection head into one detector."""

import torch.nn as nn

from models.backbone import BaselineBackbone, ResNet18LikeBackbone
from models.head import DetectionHead


class YOLOv0Baseline(nn.Module):
    """Build the configured yolov0 detector variant from a common entrypoint."""

    def __init__(
        self,
        num_classes=80,
        model_name="baseline_cnn",
        width_mult=1.0,
        depth_mult=1.0,
        use_residual=False,
        num_boxes=1,
    ):
        super().__init__()
        self.backbone = self._build_backbone(model_name, width_mult, depth_mult, use_residual)
        self.head = DetectionHead(
            in_channels=self.backbone.out_channels,
            num_classes=num_classes,
            width_mult=width_mult,
            num_boxes=num_boxes,
        )

    @staticmethod
    def _build_backbone(model_name, width_mult, depth_mult, use_residual):
        """Select the backbone variant while keeping the head interface stable."""
        if model_name in {"baseline_cnn", "deep_cnn", "residual_small", "deep_residual"}:
            return BaselineBackbone(
                width_mult=width_mult,
                depth_mult=depth_mult,
                use_residual=use_residual,
            )
        if model_name == "resnet18_like":
            return ResNet18LikeBackbone(
                width_mult=width_mult,
                depth_mult=depth_mult,
            )
        raise ValueError(f"Unsupported model_name: {model_name}")

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)
