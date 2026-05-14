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
    augmentation_cfg = config.get("augmentation", {})

    grid_sizes = data_cfg.get("grid_sizes", "n/a")
    feature_levels = parse_string_list(model_cfg.get("feature_levels"))
    per_level_anchor_lines = [
        f"anchors_{level} = {model_cfg.get(f'anchors_{level}', 'n/a')}"
        for level in feature_levels
    ]

    return [
        f"run_name = {logging_cfg['run_name']}",
        f"train_manifest = {data_cfg['train_manifest']}",
        f"val_manifest = {data_cfg['val_manifest']}",
        f"packing_format = {data_cfg['packing_format']}",
        f"packed_root = {data_cfg.get('packed_root', 'n/a')}",
        f"packed_chunk_size = {data_cfg.get('packed_chunk_size', 'n/a')}",
        f"packed_cache_size = {data_cfg.get('packed_cache_size', 'n/a')}",
        f"image_size = {data_cfg['image_size']}",
        f"grid_size = {data_cfg.get('grid_size', 'n/a')}",
        f"grid_sizes = {grid_sizes}",
        f"num_classes = {data_cfg['num_classes']}",
        f"train_max_samples = {data_cfg['train_max_samples']}",
        f"val_max_samples = {data_cfg['val_max_samples']}",
        f"model_name = {model_cfg['name']}",
        f"width_mult = {model_cfg['width_mult']}",
        f"depth_mult = {model_cfg['depth_mult']}",
        f"use_residual = {model_cfg['use_residual']}",
        f"head_type = {model_cfg.get('head_type', 'shared')}",
        f"neck_type = {model_cfg.get('neck_type', 'none')}",
        f"num_boxes = {model_cfg.get('num_boxes', 1)}",
        f"multiscale = {model_cfg.get('multiscale', False)}",
        f"feature_levels = {model_cfg.get('feature_levels', 'n/a')}",
        f"anchors = {model_cfg.get('anchors', 'n/a')}",
        f"anchor_positive_iou = {model_cfg.get('anchor_positive_iou', 'n/a')}",
        f"anchor_ignore_iou = {model_cfg.get('anchor_ignore_iou', 'n/a')}",
        f"anchor_match_metric = {model_cfg.get('anchor_match_metric', 'iou')}",
        f"anchor_shape_ratio = {model_cfg.get('anchor_shape_ratio', 'n/a')}",
        f"anchor_ignore_shape_ratio = {model_cfg.get('anchor_ignore_shape_ratio', 'n/a')}",
        f"box_parameterization = {model_cfg.get('box_parameterization', 'legacy')}",
        f"box_type = {loss_cfg['box_type']}",
        f"cls_type = {loss_cfg['cls_type']}",
        f"use_objectness = {loss_cfg['use_objectness']}",
        f"iou_loss = {loss_cfg['iou_loss']}",
        f"soft_objectness_target = {loss_cfg.get('soft_objectness_target', 'hard')}",
        f"soft_objectness_min = {loss_cfg.get('soft_objectness_min', 0.0)}",
        f"soft_classification_target = {loss_cfg.get('soft_classification_target', 'hard')}",
        f"cls_loss_mode = {loss_cfg.get('cls_loss_mode', 'bce')}",
        f"varifocal_alpha = {loss_cfg.get('varifocal_alpha', 'n/a')}",
        f"varifocal_gamma = {loss_cfg.get('varifocal_gamma', 'n/a')}",
        f"assignment_strategy = {loss_cfg.get('assignment_strategy', 'static')}",
        f"dynamic_topk = {loss_cfg.get('dynamic_topk', 'n/a')}",
        f"dynamic_center_radius = {loss_cfg.get('dynamic_center_radius', 'n/a')}",
        f"dynamic_box_cost = {loss_cfg.get('dynamic_box_cost', 'n/a')}",
        f"dynamic_cls_cost = {loss_cfg.get('dynamic_cls_cost', 'n/a')}",
        f"dynamic_ignore_iou = {loss_cfg.get('dynamic_ignore_iou', 'n/a')}",
        f"dynamic_anchor_shape_cost = {loss_cfg.get('dynamic_anchor_shape_cost', 'n/a')}",
        f"scale_assignment = {loss_cfg.get('scale_assignment', 'all')}",
        f"scale_area_threshold = {loss_cfg.get('scale_area_threshold', 'n/a')}",
        f"scale_area_thresholds = {loss_cfg.get('scale_area_thresholds', 'n/a')}",
        f"scale_loss_weights = {loss_cfg.get('scale_loss_weights', 'n/a')}",
        f"lambda_obj = {loss_cfg.get('lambda_obj', 'n/a')}",
        f"lambda_noobj = {loss_cfg.get('lambda_noobj', 'n/a')}",
        f"optimizer = {train_cfg['optimizer']}",
        f"batch_size = {train_cfg['batch_size']}",
        f"epochs = {train_cfg['epochs']}",
        f"lr = {train_cfg['lr']}",
        f"min_lr = {train_cfg['min_lr']}",
        f"weight_decay = {train_cfg['weight_decay']}",
        f"scheduler = {train_cfg['scheduler']}",
        f"num_workers = {train_cfg['num_workers']}",
        f"use_data_parallel = {train_cfg['use_data_parallel']}",
        f"max_steps_per_epoch = {train_cfg['max_steps_per_epoch']}",
        f"max_val_steps = {train_cfg['max_val_steps']}",
        f"val_interval_epochs = {config['evaluation']['val_interval_epochs']}",
        f"vis_interval_epochs = {config['visualization'].get('vis_interval_epochs', 'n/a')}",
        f"vis_max_samples = {config['visualization'].get('max_samples', 'n/a')}",
        f"eval_score_threshold = {config['evaluation'].get('score_threshold', 'n/a')}",
        f"eval_score_alpha = {config['evaluation'].get('score_alpha', 1.0)}",
        f"eval_score_beta = {config['evaluation'].get('score_beta', 1.0)}",
        f"vis_score_threshold = {config['visualization'].get('score_threshold', 'n/a')}",
        f"vis_score_alpha = {config['visualization'].get('score_alpha', 1.0)}",
        f"vis_score_beta = {config['visualization'].get('score_beta', 1.0)}",
        f"augmentation_enabled = {augmentation_cfg.get('enabled', False)}",
        f"augmentation_horizontal_flip_p = {augmentation_cfg.get('horizontal_flip_p', 'n/a')}",
        f"augmentation_color_jitter_p = {augmentation_cfg.get('color_jitter_p', 'n/a')}",
        f"augmentation_affine_p = {augmentation_cfg.get('affine_p', 'n/a')}",
        f"augmentation_brightness = {augmentation_cfg.get('brightness', 'n/a')}",
        f"augmentation_contrast = {augmentation_cfg.get('contrast', 'n/a')}",
        f"augmentation_saturation = {augmentation_cfg.get('saturation', 'n/a')}",
        f"augmentation_degrees = {augmentation_cfg.get('degrees', 'n/a')}",
        f"augmentation_translate = {augmentation_cfg.get('translate', 'n/a')}",
        f"augmentation_scale_min = {augmentation_cfg.get('scale_min', 'n/a')}",
        f"augmentation_scale_max = {augmentation_cfg.get('scale_max', 'n/a')}",
        f"augmentation_shear = {augmentation_cfg.get('shear', 'n/a')}",
        f"augmentation_scale_jitter_p = {augmentation_cfg.get('scale_jitter_p', 'n/a')}",
        f"augmentation_scale_jitter_min = {augmentation_cfg.get('scale_jitter_min', 'n/a')}",
        f"augmentation_scale_jitter_max = {augmentation_cfg.get('scale_jitter_max', 'n/a')}",
        f"augmentation_blur_p = {augmentation_cfg.get('blur_p', 'n/a')}",
        f"augmentation_blur_kernel_size = {augmentation_cfg.get('blur_kernel_size', 'n/a')}",
        f"augmentation_blur_sigma_min = {augmentation_cfg.get('blur_sigma_min', 'n/a')}",
        f"augmentation_blur_sigma_max = {augmentation_cfg.get('blur_sigma_max', 'n/a')}",
        f"augmentation_noise_p = {augmentation_cfg.get('noise_p', 'n/a')}",
        f"augmentation_noise_std = {augmentation_cfg.get('noise_std', 'n/a')}",
    ] + per_level_anchor_lines


def parse_anchor_string(raw: str | None) -> list[tuple[float, float]]:
    """Parse a compact `w,h;w,h;...` anchor string into float pairs."""
    if raw is None:
        return []
    text = str(raw).strip()
    if not text:
        return []

    anchors = []
    for item in text.split(";"):
        item = item.strip()
        if not item:
            continue
        width, height = item.split(",", 1)
        anchors.append((float(width.strip()), float(height.strip())))
    return anchors


def parse_int_list(raw: str | None) -> list[int]:
    """Parse a compact comma-separated integer list."""
    if raw is None:
        return []
    text = str(raw).strip()
    if not text:
        return []
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def parse_string_list(raw: str | None) -> list[str]:
    """Parse a compact comma-separated string list."""
    if raw is None:
        return []
    text = str(raw).strip()
    if not text:
        return []
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_float_list(raw: str | None) -> list[float]:
    """Parse a compact comma-separated float list."""
    if raw is None:
        return []
    text = str(raw).strip()
    if not text:
        return []
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def parse_anchor_map(model_cfg: dict, feature_levels: list[str]) -> dict[str, list[tuple[float, float]]]:
    """Parse per-level anchor strings such as `anchors_p4` and `anchors_p5`."""
    anchor_map: dict[str, list[tuple[float, float]]] = {}
    for level in feature_levels:
        level_anchors = parse_anchor_string(model_cfg.get(f"anchors_{level}"))
        if level_anchors:
            anchor_map[level] = level_anchors
    return anchor_map


def parse_float_map(raw: str | None) -> dict[str, float]:
    """Parse `name:value;name:value` into a float dictionary."""
    if raw is None:
        return {}
    text = str(raw).strip()
    if not text:
        return {}
    result = {}
    for item in text.split(";"):
        item = item.strip()
        if not item:
            continue
        key, value = item.split(":", 1)
        result[key.strip()] = float(value.strip())
    return result
