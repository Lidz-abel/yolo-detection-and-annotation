"""Detection head for the first yolov0 baseline detector."""

import torch.nn as nn

from models.common import ConvBNAct, make_divisible


class DetectionHead(nn.Module):
    """Map backbone features to per-grid class logits and box regression values."""

    def __init__(self, in_channels, num_classes=80, width_mult=1.0):
        super().__init__()
        hidden_channels = make_divisible(256 * width_mult)
        self.num_classes = num_classes
        self.pred_dim = num_classes + 4
        self.head = nn.Sequential(
            ConvBNAct(in_channels, hidden_channels, kernel_size=3, stride=1),
            nn.Conv2d(hidden_channels, self.pred_dim, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x = self.head(x)
        return x.permute(0, 2, 3, 1).contiguous()
