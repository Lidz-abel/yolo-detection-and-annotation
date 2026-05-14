"""Small modeling helpers used by the first yolov0 baseline."""

import torch


def count_parameters(model):
    """Return total and trainable parameter counts for experiment tracking."""
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return {"total": total, "trainable": trainable}


def describe_model_output(model, image_size, device):
    """Run one dummy forward pass to capture the detector output shape."""
    model.eval()
    with torch.no_grad():
        dummy = torch.randn(1, 3, image_size, image_size, device=device)
        pred = model(dummy)
    if isinstance(pred, dict):
        return {name: tuple(value.shape) for name, value in pred.items()}
    return tuple(pred.shape)
