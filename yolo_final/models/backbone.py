"""Backbone variants used in the controlled yolov0 structure experiments."""

import torch.nn as nn

from models.common import BasicResidualBlock, ConvBNAct, ResidualBlock, make_divisible


def _repeat_count(depth_mult: float, base: int = 2) -> int:
    """Convert one nominal repeat count into the configured stage depth."""
    return max(1, int(round(base * depth_mult)))


def _make_baseline_stage(in_channels, out_channels, repeats, use_residual):
    """Build one baseline stage with optional residual refinement."""
    blocks = [ConvBNAct(in_channels, out_channels, kernel_size=3, stride=1)]
    for _ in range(repeats - 1):
        blocks.append(ConvBNAct(out_channels, out_channels, kernel_size=3, stride=1))
    if use_residual:
        blocks.append(ResidualBlock(out_channels))
    return nn.Sequential(*blocks)


class BaselineBackbone(nn.Module):
    """Produce a fixed grid feature map suitable for the initial detector head."""

    def __init__(self, width_mult=1.0, depth_mult=1.0, use_residual=False):
        super().__init__()
        stage_channels = [
            make_divisible(32 * width_mult),
            make_divisible(64 * width_mult),
            make_divisible(128 * width_mult),
            make_divisible(256 * width_mult),
            make_divisible(384 * width_mult),
            make_divisible(512 * width_mult),
        ]

        layers = []
        in_channels = 3
        for stage_index, out_channels in enumerate(stage_channels):
            layers.append(ConvBNAct(in_channels, out_channels, kernel_size=3, stride=1))
            for _ in range(_repeat_count(depth_mult, 2) - 1):
                layers.append(ConvBNAct(out_channels, out_channels, kernel_size=3, stride=1))
            if use_residual and stage_index >= 2:
                layers.append(ResidualBlock(out_channels))
            if stage_index < len(stage_channels) - 1:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = out_channels

        self.features = nn.Sequential(*layers)
        self.out_channels = stage_channels[-1]

    def forward(self, x):
        return self.features(x)


class ResNet18LikeBackbone(nn.Module):
    """A more standard residual backbone with ResNet-18 style stage layout."""

    def __init__(self, width_mult=1.0, depth_mult=1.0):
        super().__init__()
        stem_channels = make_divisible(64 * width_mult)
        stage_channels = [
            make_divisible(64 * width_mult),
            make_divisible(128 * width_mult),
            make_divisible(256 * width_mult),
            make_divisible(512 * width_mult),
        ]

        self.stem = nn.Sequential(
            ConvBNAct(3, stem_channels, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = self._make_stage(stem_channels, stage_channels[0], _repeat_count(depth_mult, 2), stride=1)
        self.layer2 = self._make_stage(stage_channels[0], stage_channels[1], _repeat_count(depth_mult, 2), stride=2)
        self.layer3 = self._make_stage(stage_channels[1], stage_channels[2], _repeat_count(depth_mult, 2), stride=2)
        self.layer4 = self._make_stage(stage_channels[2], stage_channels[3], _repeat_count(depth_mult, 2), stride=2)
        self.out_channels = stage_channels[-1]

    @staticmethod
    def _make_stage(in_channels, out_channels, num_blocks, stride):
        """Build one residual stage with one optional downsampling block."""
        blocks = [BasicResidualBlock(in_channels, out_channels, stride=stride)]
        for _ in range(num_blocks - 1):
            blocks.append(BasicResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class MultiScaleBaselineBackbone(nn.Module):
    """Expose configured multi-scale features for the Stage-D detector."""

    def __init__(self, width_mult=1.0, depth_mult=1.0, use_residual=True, feature_levels=None):
        super().__init__()
        self.include_p3 = "p3" in (feature_levels or [])
        stage_channels = [
            make_divisible(32 * width_mult),
            make_divisible(64 * width_mult),
            make_divisible(128 * width_mult),
            make_divisible(256 * width_mult),
            make_divisible(384 * width_mult),
            make_divisible(512 * width_mult),
        ]

        self.stage1 = _make_baseline_stage(3, stage_channels[0], _repeat_count(depth_mult, 2), use_residual=False)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stage2 = _make_baseline_stage(stage_channels[0], stage_channels[1], _repeat_count(depth_mult, 2), use_residual=False)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stage3 = _make_baseline_stage(stage_channels[1], stage_channels[2], _repeat_count(depth_mult, 2), use_residual=use_residual)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stage4 = _make_baseline_stage(stage_channels[2], stage_channels[3], _repeat_count(depth_mult, 2), use_residual=use_residual)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stage5 = _make_baseline_stage(stage_channels[3], stage_channels[4], _repeat_count(depth_mult, 2), use_residual=use_residual)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stage6 = _make_baseline_stage(stage_channels[4], stage_channels[5], _repeat_count(depth_mult, 2), use_residual=use_residual)
        self.out_channels = {"p4": stage_channels[3], "p5": stage_channels[5]}
        if self.include_p3:
            self.out_channels = {"p3": stage_channels[3], **self.out_channels}

    def forward(self, x):
        x = self.pool1(self.stage1(x))
        x = self.pool2(self.stage2(x))
        x = self.pool3(self.stage3(x))
        x = self.stage4(x)
        p3 = x
        p4 = self.pool4(x)
        x = self.pool5(self.stage5(p4))
        p5 = self.stage6(x)
        features = {"p4": p4, "p5": p5}
        if self.include_p3:
            features = {"p3": p3, **features}
        return features
