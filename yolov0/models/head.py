"""Detection head for the yolov0 detector variants."""

import torch.nn as nn

from models.common import ConvBNAct, make_divisible


class DetectionHead(nn.Module):
    """Map backbone features to per-grid box, objectness, and class predictions."""

    def __init__(self, in_channels, num_classes=80, width_mult=1.0, num_boxes=1):
        super().__init__()
        hidden_channels = make_divisible(256 * width_mult)
        self.num_classes = num_classes
        self.num_boxes = num_boxes
        self.pred_dim = num_boxes * (num_classes + 5)
        self.head = nn.Sequential(
            ConvBNAct(in_channels, hidden_channels, kernel_size=3, stride=1),
            nn.Conv2d(hidden_channels, self.pred_dim, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x = self.head(x)
        return x.permute(0, 2, 3, 1).contiguous()
