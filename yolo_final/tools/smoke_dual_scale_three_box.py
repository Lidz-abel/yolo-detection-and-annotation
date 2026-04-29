"""Smoke-check the dual-scale three-box configuration on one real sample."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from data.detection_dataset import DetectionDataset
from losses.yolo_loss import YOLOLoss
from models.detector import YOLOv0Baseline
from utils.config import load_config, parse_anchor_map, parse_anchor_string, parse_float_map, parse_int_list, parse_string_list


def parse_args():
    """Parse the config path used by the smoke check."""
    parser = argparse.ArgumentParser(description="Smoke-check one dual-scale three-box config.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "dual_scale_three_box_formal.toml"),
        help="Config file to smoke-check.",
    )
    return parser.parse_args()


def main():
    """Run one minimal forward and loss pass to verify the code path is wired."""
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    data_cfg = config["data"]
    model_cfg = config["model"]
    loss_cfg = config["loss"]

    grid_sizes = parse_int_list(data_cfg.get("grid_sizes"))
    if not grid_sizes:
        grid_sizes = [int(data_cfg["grid_size"])]
    feature_levels = parse_string_list(model_cfg.get("feature_levels"))
    if not feature_levels:
        feature_levels = [f"scale_{index}" for index in range(len(grid_sizes))]

    anchors = parse_anchor_string(model_cfg.get("anchors"))
    anchors_by_level = parse_anchor_map(model_cfg, feature_levels)
    default_grid_size = int(data_cfg.get("grid_size", grid_sizes[0]))
    num_boxes = int(model_cfg.get("num_boxes", 1))

    dataset = DetectionDataset(
        manifest_path=data_cfg["val_manifest"],
        image_size=int(data_cfg["image_size"]),
        grid_size=default_grid_size,
        grid_sizes=grid_sizes,
        feature_levels=feature_levels,
        num_classes=int(data_cfg["num_classes"]),
        num_boxes=num_boxes,
        anchors=anchors,
        anchors_by_level=anchors_by_level,
        anchor_positive_iou=float(model_cfg.get("anchor_positive_iou", 0.25)),
        anchor_ignore_iou=float(model_cfg.get("anchor_ignore_iou", 0.5)),
        anchor_match_metric=str(model_cfg.get("anchor_match_metric", "iou")),
        anchor_shape_ratio=float(model_cfg.get("anchor_shape_ratio", 4.0)),
        anchor_ignore_shape_ratio=model_cfg.get("anchor_ignore_shape_ratio"),
        max_samples=1,
        packing_format=str(data_cfg.get("packing_format", "raw")),
        packed_root=data_cfg.get("packed_root"),
        packed_chunk_size=data_cfg.get("packed_chunk_size"),
        packed_cache_size=int(data_cfg.get("packed_cache_size", 4)),
        require_packed=str(data_cfg.get("packing_format", "raw")).lower() == "pt",
    )

    image_tensor, target = dataset[0]
    image_tensor = image_tensor.unsqueeze(0)
    target["multiscale_targets"] = {
        scale_name: {
            key: value.unsqueeze(0)
            for key, value in scale_targets.items()
        }
        for scale_name, scale_targets in target["multiscale_targets"].items()
    }
    target["boxes"] = [target["boxes"]]
    target["labels"] = [target["labels"]]
    target["original_size"] = target["original_size"].unsqueeze(0)
    target["resized_size"] = target["resized_size"].unsqueeze(0)

    model = YOLOv0Baseline(
        num_classes=int(data_cfg["num_classes"]),
        model_name=str(model_cfg["name"]),
        width_mult=float(model_cfg["width_mult"]),
        depth_mult=float(model_cfg["depth_mult"]),
        use_residual=bool(model_cfg["use_residual"]),
        num_boxes=num_boxes,
        head_type=str(model_cfg.get("head_type", "shared")),
        neck_type=str(model_cfg.get("neck_type", "none")),
        feature_levels=feature_levels,
    )
    criterion = YOLOLoss(
        num_classes=int(data_cfg["num_classes"]),
        lambda_box=float(loss_cfg["lambda_box"]),
        lambda_obj=float(loss_cfg.get("lambda_obj", 1.0)),
        lambda_noobj=float(loss_cfg.get("lambda_noobj", 0.5)),
        lambda_cls=float(loss_cfg["lambda_cls"]),
        num_boxes=num_boxes,
        anchors=anchors,
        anchors_by_level=anchors_by_level,
        box_parameterization=str(model_cfg.get("box_parameterization", "legacy")),
        soft_objectness_target=str(loss_cfg.get("soft_objectness_target", "hard")),
        soft_objectness_min=float(loss_cfg.get("soft_objectness_min", 0.0)),
        soft_classification_target=str(loss_cfg.get("soft_classification_target", "hard")),
        cls_loss_mode=str(loss_cfg.get("cls_loss_mode", "bce")),
        assignment_strategy=str(loss_cfg.get("assignment_strategy", "static")),
        dynamic_topk=int(loss_cfg.get("dynamic_topk", 2)),
        dynamic_center_radius=int(loss_cfg.get("dynamic_center_radius", 1)),
        dynamic_box_cost=float(loss_cfg.get("dynamic_box_cost", 3.0)),
        dynamic_cls_cost=float(loss_cfg.get("dynamic_cls_cost", 1.0)),
        dynamic_ignore_iou=float(loss_cfg.get("dynamic_ignore_iou", 0.5)),
        dynamic_anchor_shape_cost=float(loss_cfg.get("dynamic_anchor_shape_cost", 0.0)),
        scale_assignment=str(loss_cfg.get("scale_assignment", "all")),
        scale_area_threshold=float(loss_cfg.get("scale_area_threshold", 0.2)),
        scale_loss_weights=parse_float_map(loss_cfg.get("scale_loss_weights")),
        feature_levels=feature_levels,
    )

    with torch.no_grad():
        pred = model(image_tensor)
        loss_dict = criterion(pred, target)

    print("smoke config:", config_path)
    print("feature_levels:", feature_levels)
    print("grid_sizes:", grid_sizes)
    print("anchors_by_level:", anchors_by_level)
    print("dataset storage:", dataset.storage_mode)
    if dataset.packed_summary is not None:
        print("packed index:", dataset.packed_summary["index_path"])
    for scale_name, scale_pred in pred.items():
        print(f"{scale_name} output shape:", tuple(scale_pred.shape))
    print("loss total:", float(loss_dict["total_loss"].item()))
    print("loss box:", float(loss_dict["loss_box"].item()))
    print("loss obj:", float(loss_dict["loss_obj"].item()))
    print("loss cls:", float(loss_dict["loss_cls"].item()))


if __name__ == "__main__":
    main()
