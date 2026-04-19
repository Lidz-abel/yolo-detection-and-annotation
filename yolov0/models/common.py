"""Reusable convolution and residual blocks for yolov0 backbones."""

import torch.nn as nn


def make_divisible(value, divisor=8):
    """Round channel counts to hardware-friendly multiples."""
    return max(divisor, int((value + divisor / 2) // divisor * divisor))


class ConvBNAct(nn.Module):
    """A standard conv-bn-relu block used throughout the baseline model."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ConvBN(nn.Module):
    """A conv-bn block without activation for residual branch endings."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    """A lightweight residual block used when residual connections are enabled."""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = ConvBNAct(channels, channels, kernel_size=3, stride=1)
        self.conv2 = ConvBNAct(channels, channels, kernel_size=3, stride=1)

    def forward(self, x):
        return x + self.conv2(self.conv1(x))


class BasicResidualBlock(nn.Module):
    """A more standard ResNet-style basic block with post-add activation."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = ConvBNAct(in_channels, out_channels, kernel_size=3, stride=stride)
        self.conv2 = ConvBN(out_channels, out_channels, kernel_size=3, stride=1)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = ConvBN(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)
        else:
            self.shortcut = nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + identity
        return self.relu(x)
