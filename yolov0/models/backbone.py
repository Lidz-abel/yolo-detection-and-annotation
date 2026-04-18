"""Baseline backbone that can gradually evolve via depth and residual switches."""

import torch.nn as nn

from models.common import ConvBNAct, ResidualBlock, make_divisible


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
