"""Backbone variants used in the controlled detection structure experiments."""

import torch
import torch.nn as nn
import torch.nn.functional as F

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


class ViTDualScaleBackbone(nn.Module):
    """A compact ViT backbone that preserves the detector's p4/p5 feature contract."""

    def __init__(
        self,
        width_mult=1.0,
        depth_mult=1.0,
        image_size=416,
        patch_size=16,
        feature_levels=None,
    ):
        super().__init__()
        self.feature_levels = feature_levels or ["p4", "p5"]
        if self.feature_levels != ["p4", "p5"]:
            raise ValueError("ViTDualScaleBackbone currently supports feature_levels='p4,p5'.")
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size for ViTDualScaleBackbone.")

        embed_dim = make_divisible(384 * width_mult)
        depth = max(2, int(round(6 * depth_mult)))
        num_heads = self._pick_num_heads(embed_dim)
        mlp_dim = int(embed_dim * 4)
        self.patch_size = int(patch_size)
        self.base_grid_size = int(image_size) // int(patch_size)
        self.embed_dim = embed_dim

        self.patch_embed = nn.Conv2d(
            3,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim, self.base_grid_size, self.base_grid_size))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.final_norm = nn.LayerNorm(embed_dim)

        self.p4_refine = ConvBNAct(embed_dim, embed_dim, kernel_size=3, stride=1)
        self.p5_downsample = ConvBNAct(embed_dim, embed_dim, kernel_size=3, stride=2)
        self.out_channels = {"p4": embed_dim, "p5": embed_dim}

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    @staticmethod
    def _pick_num_heads(embed_dim):
        """Choose a valid attention-head count for the configured embedding width."""
        for heads in (12, 8, 6, 4, 3, 2, 1):
            if embed_dim % heads == 0:
                return heads
        return 1

    def _position_embedding(self, height, width):
        """Interpolate positional embeddings if the input grid differs from 416/16."""
        if height == self.base_grid_size and width == self.base_grid_size:
            return self.pos_embed
        return F.interpolate(
            self.pos_embed,
            size=(height, width),
            mode="bicubic",
            align_corners=False,
        )

    def forward(self, x):
        tokens_2d = self.patch_embed(x)
        batch_size, channels, height, width = tokens_2d.shape
        tokens_2d = tokens_2d + self._position_embedding(height, width)
        tokens = tokens_2d.flatten(2).transpose(1, 2)
        tokens = self.encoder(tokens)
        tokens = self.final_norm(tokens)
        p4 = tokens.transpose(1, 2).reshape(batch_size, channels, height, width)
        p4 = self.p4_refine(p4)
        p5 = self.p5_downsample(p4)
        return {"p4": p4, "p5": p5}


class HybridViTDetectionBackbone(nn.Module):
    """Detection-oriented ViT backbone with a high-resolution p3 branch."""

    def __init__(
        self,
        width_mult=1.0,
        depth_mult=1.0,
        image_size=416,
        feature_levels=None,
    ):
        super().__init__()
        self.feature_levels = feature_levels or ["p3", "p4", "p5"]
        supported_levels = {"p3", "p4", "p5"}
        unknown_levels = set(self.feature_levels) - supported_levels
        if unknown_levels:
            raise ValueError(f"HybridViTDetectionBackbone unsupported feature levels: {sorted(unknown_levels)}")
        if "p4" not in self.feature_levels or "p5" not in self.feature_levels:
            raise ValueError("HybridViTDetectionBackbone expects at least p4 and p5 feature levels.")
        if image_size % 16 != 0:
            raise ValueError("image_size must be divisible by 16 for HybridViTDetectionBackbone.")

        stem_channels = make_divisible(64 * width_mult)
        mid_channels = make_divisible(128 * width_mult)
        p3_channels = make_divisible(256 * width_mult)
        embed_dim = make_divisible(256 * width_mult)
        p5_channels = make_divisible(512 * width_mult)
        depth = max(4, int(round(8 * depth_mult)))
        num_heads = self._pick_num_heads(embed_dim)
        mlp_dim = int(embed_dim * 4)

        self.base_grid_size = int(image_size) // 16
        self.embed_dim = embed_dim

        self.stem = nn.Sequential(
            ConvBNAct(3, stem_channels, kernel_size=3, stride=2),
            ConvBNAct(stem_channels, mid_channels, kernel_size=3, stride=2),
            ConvBNAct(mid_channels, p3_channels, kernel_size=3, stride=2),
            ConvBNAct(p3_channels, p3_channels, kernel_size=3, stride=1),
        )
        self.p4_embed = ConvBNAct(p3_channels, embed_dim, kernel_size=3, stride=2)
        self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim, self.base_grid_size, self.base_grid_size))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.final_norm = nn.LayerNorm(embed_dim)
        self.p4_refine = ConvBNAct(embed_dim, embed_dim, kernel_size=3, stride=1)
        self.p5_downsample = ConvBNAct(embed_dim, p5_channels, kernel_size=3, stride=2)
        self.p5_refine = ConvBNAct(p5_channels, p5_channels, kernel_size=3, stride=1)

        self.out_channels = {"p4": embed_dim, "p5": p5_channels}
        if "p3" in self.feature_levels:
            self.out_channels = {"p3": p3_channels, **self.out_channels}

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    @staticmethod
    def _pick_num_heads(embed_dim):
        """Choose a valid attention-head count for the configured embedding width."""
        for heads in (12, 8, 6, 4, 3, 2, 1):
            if embed_dim % heads == 0:
                return heads
        return 1

    def _position_embedding(self, height, width):
        """Interpolate positional embeddings if the input grid differs from 416/16."""
        if height == self.base_grid_size and width == self.base_grid_size:
            return self.pos_embed
        return F.interpolate(
            self.pos_embed,
            size=(height, width),
            mode="bicubic",
            align_corners=False,
        )

    def forward(self, x):
        p3 = self.stem(x)
        p4_tokens_2d = self.p4_embed(p3)
        batch_size, channels, height, width = p4_tokens_2d.shape
        p4_tokens_2d = p4_tokens_2d + self._position_embedding(height, width)
        tokens = p4_tokens_2d.flatten(2).transpose(1, 2)
        tokens = self.encoder(tokens)
        tokens = self.final_norm(tokens)
        p4 = tokens.transpose(1, 2).reshape(batch_size, channels, height, width)
        p4 = self.p4_refine(p4)
        p5 = self.p5_refine(self.p5_downsample(p4))

        features = {"p4": p4, "p5": p5}
        if "p3" in self.feature_levels:
            features = {"p3": p3, **features}
        return features
