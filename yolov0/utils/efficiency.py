"""Model efficiency helpers for FLOPs estimation and inference benchmarking."""

from __future__ import annotations

import time

import torch
import torch.nn as nn


def estimate_flops(model: nn.Module, image_size: int, device: torch.device) -> dict[str, float]:
    """Estimate Conv2d and Linear FLOPs with one dummy forward pass."""
    handles = []
    flops = {"conv2d": 0.0, "linear": 0.0}

    def conv_hook(module: nn.Conv2d, inputs, output):
        batch_size = output.shape[0]
        out_h = output.shape[2]
        out_w = output.shape[3]
        kernel_h, kernel_w = module.kernel_size
        in_channels = module.in_channels
        out_channels = module.out_channels
        groups = module.groups
        kernel_mul = kernel_h * kernel_w * (in_channels / groups)
        macs = batch_size * out_h * out_w * out_channels * kernel_mul
        flops["conv2d"] += 2.0 * macs

    def linear_hook(module: nn.Linear, inputs, output):
        batch_size = inputs[0].shape[0]
        macs = batch_size * module.in_features * module.out_features
        flops["linear"] += 2.0 * macs

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))

    dummy = torch.randn(1, 3, image_size, image_size, device=device)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(dummy)
    if was_training:
        model.train()

    for handle in handles:
        handle.remove()

    total = flops["conv2d"] + flops["linear"]
    return {
        "conv2d_flops": flops["conv2d"],
        "linear_flops": flops["linear"],
        "total_flops": total,
        "total_gflops": total / 1e9,
    }


def benchmark_fps(
    model: nn.Module,
    image_size: int,
    device: torch.device,
    batch_size: int = 1,
    warmup_iters: int = 20,
    measure_iters: int = 100,
) -> dict[str, float]:
    """Measure forward-only throughput on one device with dummy inputs."""
    was_training = model.training
    model.eval()
    dummy = torch.randn(batch_size, 3, image_size, image_size, device=device)

    with torch.no_grad():
        for _ in range(warmup_iters):
            model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        start = time.perf_counter()
        for _ in range(measure_iters):
            model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        duration = time.perf_counter() - start

    if was_training:
        model.train()

    total_images = batch_size * measure_iters
    images_per_second = total_images / max(duration, 1e-9)
    milliseconds_per_batch = (duration / measure_iters) * 1000.0
    return {
        "batch_size": float(batch_size),
        "warmup_iters": float(warmup_iters),
        "measure_iters": float(measure_iters),
        "duration_seconds": duration,
        "images_per_second": images_per_second,
        "milliseconds_per_batch": milliseconds_per_batch,
    }
