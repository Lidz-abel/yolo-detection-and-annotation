"""Lightweight neck modules used by the dual-scale detector variants."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common import ConvBNAct, make_divisible


class DualScaleFPNPANLite(nn.Module):
    """Fuse `p4` and `p5` with a lightweight top-down and bottom-up path."""

    def __init__(self, in_channels: dict[str, int], width_mult: float = 1.0):
        super().__init__()
        if "p4" not in in_channels or "p5" not in in_channels:
            raise ValueError("DualScaleFPNPANLite expects `p4` and `p5` feature maps.")

        p4_channels = int(in_channels["p4"])
        p5_channels = int(in_channels["p5"])
        hidden_p4 = make_divisible(p4_channels * width_mult)
        hidden_p5 = make_divisible(p5_channels * width_mult)

        # Top-down path: align p5 channels, upsample, and merge into p4.
        self.p5_lateral = ConvBNAct(p5_channels, hidden_p4, kernel_size=1, stride=1, padding=0)
        self.p4_reduce = ConvBNAct(p4_channels, hidden_p4, kernel_size=1, stride=1, padding=0)
        self.p4_fuse = ConvBNAct(hidden_p4, hidden_p4, kernel_size=3, stride=1)

        # Bottom-up refinement: project fused p4 back into the coarser stream.
        self.p4_downsample = ConvBNAct(hidden_p4, hidden_p5, kernel_size=3, stride=2)
        self.p5_reduce = ConvBNAct(p5_channels, hidden_p5, kernel_size=1, stride=1, padding=0)
        self.p5_fuse = ConvBNAct(hidden_p5, hidden_p5, kernel_size=3, stride=1)

        self.out_channels = {"p4": hidden_p4, "p5": hidden_p5}

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Return fused `p4` and `p5` features with the same spatial scales."""
        p4 = features["p4"]
        p5 = features["p5"]

        p5_td = self.p5_lateral(p5)
        p5_up = F.interpolate(p5_td, size=p4.shape[-2:], mode="nearest")
        p4_td = self.p4_fuse(self.p4_reduce(p4) + p5_up)

        p4_down = self.p4_downsample(p4_td)
        p5_out = self.p5_fuse(self.p5_reduce(p5) + p4_down)

        return {"p4": p4_td, "p5": p5_out}
