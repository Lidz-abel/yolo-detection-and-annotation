"""Exponential moving average helpers for detector training."""

from __future__ import annotations

import copy
import math

import torch
from torch.nn.parallel import DistributedDataParallel


def unwrap_model(model):
    """Return the underlying module when a model is wrapped by DDP."""
    if isinstance(model, DistributedDataParallel):
        return model.module
    return model


class ModelEMA:
    """Track a smoothed copy of model weights for validation and export."""

    def __init__(self, model, decay: float = 0.9998, warmup_updates: int = 2000):
        self.ema = copy.deepcopy(unwrap_model(model)).eval()
        self.decay = float(decay)
        self.warmup_updates = max(1, int(warmup_updates))
        self.updates = 0
        for parameter in self.ema.parameters():
            parameter.requires_grad_(False)

    def _decay(self) -> float:
        return self.decay * (1.0 - math.exp(-float(self.updates) / float(self.warmup_updates)))

    @torch.no_grad()
    def update(self, model) -> None:
        """Update EMA parameters from the current train model."""
        self.updates += 1
        decay = self._decay()
        model_state = unwrap_model(model).state_dict()
        ema_state = self.ema.state_dict()
        for key, ema_value in ema_state.items():
            model_value = model_state[key].detach()
            if torch.is_floating_point(ema_value):
                ema_value.mul_(decay).add_(model_value.to(dtype=ema_value.dtype), alpha=1.0 - decay)
            else:
                ema_value.copy_(model_value)

    def state_dict(self):
        return self.ema.state_dict()

    def load_state_dict(self, state_dict) -> None:
        self.ema.load_state_dict(state_dict)
