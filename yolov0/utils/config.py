from __future__ import annotations

"""Config validation helpers used by the yolov0 training entrypoints."""

from pathlib import Path

from utils.simple_toml import load_toml


REQUIRED_SECTIONS = (
    "data",
    "model",
    "loss",
    "train",
    "logging",
    "evaluation",
    "visualization",
)


def load_config(path: str | Path) -> dict:
    """Load a config file and validate that required sections exist."""
    config = load_toml(path)
    missing = [name for name in REQUIRED_SECTIONS if name not in config]
    if missing:
        raise KeyError(f"Missing required config sections: {', '.join(missing)}")
    return config


def summarize_config(config: dict) -> list[str]:
    """Create a compact text summary for result files and logs."""
    data_cfg = config["data"]
    model_cfg = config["model"]
    loss_cfg = config["loss"]
    train_cfg = config["train"]
    logging_cfg = config["logging"]

    return [
        f"run_name = {logging_cfg['run_name']}",
        f"train_manifest = {data_cfg['train_manifest']}",
        f"val_manifest = {data_cfg['val_manifest']}",
        f"packing_format = {data_cfg['packing_format']}",
        f"image_size = {data_cfg['image_size']}",
        f"grid_size = {data_cfg['grid_size']}",
        f"num_classes = {data_cfg['num_classes']}",
        f"train_max_samples = {data_cfg['train_max_samples']}",
        f"val_max_samples = {data_cfg['val_max_samples']}",
        f"model_name = {model_cfg['name']}",
        f"width_mult = {model_cfg['width_mult']}",
        f"depth_mult = {model_cfg['depth_mult']}",
        f"use_residual = {model_cfg['use_residual']}",
        f"box_type = {loss_cfg['box_type']}",
        f"cls_type = {loss_cfg['cls_type']}",
        f"use_objectness = {loss_cfg['use_objectness']}",
        f"iou_loss = {loss_cfg['iou_loss']}",
        f"optimizer = {train_cfg['optimizer']}",
        f"batch_size = {train_cfg['batch_size']}",
        f"epochs = {train_cfg['epochs']}",
        f"lr = {train_cfg['lr']}",
        f"min_lr = {train_cfg['min_lr']}",
        f"weight_decay = {train_cfg['weight_decay']}",
        f"scheduler = {train_cfg['scheduler']}",
        f"num_workers = {train_cfg['num_workers']}",
        f"use_data_parallel = {train_cfg['use_data_parallel']}",
        f"bootstrap_only = {train_cfg['bootstrap_only']}",
    ]
