"""Compose a selected backbone and detection head into one detector."""

import torch.nn as nn

from models.backbone import BaselineBackbone, MultiScaleBaselineBackbone, ResNet18LikeBackbone
from models.head import DecoupledDetectionHead, DetectionHead
from models.neck import DualScaleFPNPANLite


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
        head_type="shared",
        neck_type="none",
        feature_levels=None,
    ):
        super().__init__()
        self.feature_levels = feature_levels or ["p4", "p5"]
        self.backbone = self._build_backbone(model_name, width_mult, depth_mult, use_residual)
        self.neck = self._build_neck(
            neck_type=neck_type,
            backbone_out_channels=self.backbone.out_channels,
            width_mult=width_mult,
        )
        head_in_channels = self.neck.out_channels if self.neck is not None else self.backbone.out_channels
        self.head = self._build_head(
            head_type=head_type,
            in_channels=head_in_channels,
            num_classes=num_classes,
            width_mult=width_mult,
            num_boxes=num_boxes,
            feature_levels=self.feature_levels,
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
        if model_name in {"deep_residual_multiscale", "deep_residual_dual_scale_three_box"}:
            return MultiScaleBaselineBackbone(
                width_mult=width_mult,
                depth_mult=depth_mult,
                use_residual=use_residual,
            )
        raise ValueError(f"Unsupported model_name: {model_name}")

    @staticmethod
    def _build_neck(neck_type, backbone_out_channels, width_mult):
        """Optionally insert a lightweight multi-scale fusion neck."""
        if neck_type in {"none", "", None}:
            return None
        if isinstance(backbone_out_channels, dict) and neck_type == "fpn_pan_lite":
            return DualScaleFPNPANLite(
                in_channels=backbone_out_channels,
                width_mult=width_mult,
            )
        raise ValueError(f"Unsupported neck_type: {neck_type}")

    @staticmethod
    def _build_head(head_type, in_channels, num_classes, width_mult, num_boxes, feature_levels):
        """Select the shared or decoupled detection head from config."""
        if isinstance(in_channels, dict):
            return nn.ModuleDict(
                {
                    level: YOLOv0Baseline._build_head(
                        head_type=head_type,
                        in_channels=in_channels[level],
                        num_classes=num_classes,
                        width_mult=width_mult,
                        num_boxes=num_boxes,
                        feature_levels=feature_levels,
                    )
                    for level in feature_levels
                }
            )
        if head_type == "shared":
            return DetectionHead(
                in_channels=in_channels,
                num_classes=num_classes,
                width_mult=width_mult,
                num_boxes=num_boxes,
            )
        if head_type == "decoupled":
            return DecoupledDetectionHead(
                in_channels=in_channels,
                num_classes=num_classes,
                width_mult=width_mult,
                num_boxes=num_boxes,
            )
        raise ValueError(f"Unsupported head_type: {head_type}")

    def forward(self, x):
        features = self.backbone(x)
        if self.neck is not None:
            features = self.neck(features)
        if isinstance(features, dict):
            return {level: self.head[level](features[level]) for level in self.feature_levels}
        return self.head(features)
