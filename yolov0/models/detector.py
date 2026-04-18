"""Compose the baseline backbone and detection head into one detector."""

import torch.nn as nn

from models.backbone import BaselineBackbone
from models.head import DetectionHead


class YOLOv0Baseline(nn.Module):
    """Initial trainable detector used before fuller YOLO losses are added."""

    def __init__(self, num_classes=80, width_mult=1.0, depth_mult=1.0, use_residual=False):
        super().__init__()
        self.backbone = BaselineBackbone(
            width_mult=width_mult,
            depth_mult=depth_mult,
            use_residual=use_residual,
        )
        self.head = DetectionHead(
            in_channels=self.backbone.out_channels,
            num_classes=num_classes,
            width_mult=width_mult,
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)
