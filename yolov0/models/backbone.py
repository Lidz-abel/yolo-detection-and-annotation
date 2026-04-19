"""Backbone variants used in the controlled yolov0 structure experiments."""

import torch.nn as nn

from models.common import BasicResidualBlock, ConvBNAct, ResidualBlock, make_divisible


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

        def repeat_count(base):
            return max(1, int(round(base * depth_mult)))

        layers = []
        in_channels = 3
        for stage_index, out_channels in enumerate(stage_channels):
            layers.append(ConvBNAct(in_channels, out_channels, kernel_size=3, stride=1))
            for _ in range(repeat_count(2) - 1):
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

        def repeat_count(base):
            return max(1, int(round(base * depth_mult)))

        self.stem = nn.Sequential(
            ConvBNAct(3, stem_channels, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = self._make_stage(stem_channels, stage_channels[0], repeat_count(2), stride=1)
        self.layer2 = self._make_stage(stage_channels[0], stage_channels[1], repeat_count(2), stride=2)
        self.layer3 = self._make_stage(stage_channels[1], stage_channels[2], repeat_count(2), stride=2)
        self.layer4 = self._make_stage(stage_channels[2], stage_channels[3], repeat_count(2), stride=2)
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
