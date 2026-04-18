import torch.nn as nn

from models.detection_head import DetectionHead
from models.minimal_backbone import MinimalBackbone


class MiniYOLO(nn.Module):
    def __init__(self, num_classes=20):
        super().__init__()
        self.backbone = MinimalBackbone()
        self.head = DetectionHead(in_channels=512, num_classes=num_classes)

    def forward(self, x):
        feature = self.backbone(x)
        pred = self.head(feature)
        return pred
