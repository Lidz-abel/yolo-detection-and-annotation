"""Detection heads for the yolov0 detector variants."""

import torch.nn as nn
import torch

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


class DecoupledDetectionHead(nn.Module):
    """Split classification and box/objectness prediction into separate branches."""

    def __init__(self, in_channels, num_classes=80, width_mult=1.0, num_boxes=1):
        super().__init__()
        hidden_channels = make_divisible(256 * width_mult)
        self.num_classes = num_classes
        self.num_boxes = num_boxes
        self.pred_dim = num_boxes * (num_classes + 5)

        # A light shared stem keeps the detector interface unchanged while
        # giving the cls and reg branches their own convolutional capacity.
        self.stem = ConvBNAct(in_channels, hidden_channels, kernel_size=3, stride=1)
        self.cls_branch = nn.Sequential(
            ConvBNAct(hidden_channels, hidden_channels, kernel_size=3, stride=1),
            ConvBNAct(hidden_channels, hidden_channels, kernel_size=3, stride=1),
        )
        self.reg_branch = nn.Sequential(
            ConvBNAct(hidden_channels, hidden_channels, kernel_size=3, stride=1),
            ConvBNAct(hidden_channels, hidden_channels, kernel_size=3, stride=1),
        )
        self.cls_pred = nn.Conv2d(hidden_channels, num_boxes * num_classes, kernel_size=1, stride=1)
        self.reg_pred = nn.Conv2d(hidden_channels, num_boxes * 5, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.stem(x)
        cls_feat = self.cls_branch(x)
        reg_feat = self.reg_branch(x)

        cls_logits = self.cls_pred(cls_feat)
        reg_logits = self.reg_pred(reg_feat)

        batch_size, _, grid_h, grid_w = cls_logits.shape
        cls_logits = cls_logits.view(batch_size, self.num_boxes, self.num_classes, grid_h, grid_w)
        reg_logits = reg_logits.view(batch_size, self.num_boxes, 5, grid_h, grid_w)
        pred = torch.cat([reg_logits, cls_logits], dim=2)
        pred = pred.permute(0, 3, 4, 1, 2).contiguous()
        return pred.view(batch_size, grid_h, grid_w, self.pred_dim)
