"""Run pycocotools-based evaluation on the COCO subset of the unified data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from data.detection_dataset import DetectionDataset
from models.detector import YOLOv0Baseline
from utils.coco_eval import evaluate_coco_subset
from utils.config import load_config, parse_anchor_map, parse_anchor_string, parse_int_list, parse_string_list
from utils.efficiency import benchmark_fps, estimate_flops
from utils.modeling import count_parameters, describe_model_output


def parse_args():
    """Parse config, checkpoint, and optional manifest overrides."""
    parser = argparse.ArgumentParser(description="Evaluate one checkpoint on COCO with pycocotools.")
    parser.add_argument("--config", type=str, required=True, help="Training config used for the run.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint to evaluate.")
    parser.add_argument(
        "--manifest",
        type=str,
        default="/home/lidz/YOLO/DataSet/Unified/manifests/coco2017_val.jsonl",
        help="COCO subset manifest path.",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default="/home/lidz/YOLO/DataSet/Unified/metadata/class_maps.json",
        help="Unified metadata path containing COCO category mappings.",
    )
    parser.add_argument("--max-samples", type=int, default=0, help="Optional dataset cap.")
    parser.add_argument("--score-threshold", type=float, default=0.05, help="Decode score threshold.")
    parser.add_argument("--score-alpha", type=float, default=1.0, help="Objectness exponent used in ranking.")
    parser.add_argument("--score-beta", type=float, default=1.0, help="Classification exponent used in ranking.")
    parser.add_argument("--top-k", type=int, default=100, help="Maximum kept predictions per image.")
    parser.add_argument("--nms-iou-threshold", type=float, default=0.5, help="Class-wise NMS IoU threshold.")
    parser.add_argument("--fps-batch-size", type=int, default=1, help="Batch size used for FPS benchmark.")
    parser.add_argument("--fps-warmup-iters", type=int, default=20, help="Warmup iterations for FPS benchmark.")
    parser.add_argument("--fps-measure-iters", type=int, default=100, help="Measured iterations for FPS benchmark.")
    parser.add_argument("--output-json", type=str, default="", help="Optional JSON output path.")
    return parser.parse_args()


def build_device(device_name: str):
    """Resolve the configured device name into one torch device."""
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def load_state_dict(model, state_dict):
    """Load checkpoint weights into wrapped or plain modules uniformly."""
    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)


def main():
    """Evaluate one formal yolov0 run on the COCO subset with official tooling."""
    args = parse_args()
    config = load_config(Path(args.config).resolve())
    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["train"]
    anchors = parse_anchor_string(model_cfg.get("anchors"))
    num_boxes = int(model_cfg.get("num_boxes", 1))
    grid_sizes = parse_int_list(data_cfg.get("grid_sizes"))
    feature_levels = parse_string_list(model_cfg.get("feature_levels"))
    if not grid_sizes:
        grid_sizes = [int(data_cfg["grid_size"])]
    if not feature_levels:
        feature_levels = [f"scale_{index}" for index in range(len(grid_sizes))]
    anchors_by_level = parse_anchor_map(model_cfg, feature_levels)
    box_parameterization = str(model_cfg.get("box_parameterization", "legacy"))
    device = build_device(str(train_cfg["device"]))
    default_grid_size = int(data_cfg.get("grid_size", grid_sizes[0]))

    dataset = DetectionDataset(
        manifest_path=args.manifest,
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
        max_samples=args.max_samples,
    )

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
    ).to(device)
    if device.type == "cuda" and torch.cuda.device_count() > 1 and bool(train_cfg["use_data_parallel"]):
        model = torch.nn.DataParallel(model)

    checkpoint = torch.load(Path(args.checkpoint).resolve(), map_location=device)
    load_state_dict(model, checkpoint)

    output_shape = describe_model_output(model, int(data_cfg["image_size"]), device)
    params = count_parameters(model)
    flops = estimate_flops(model, int(data_cfg["image_size"]), device)
    fps = benchmark_fps(
        model,
        image_size=int(data_cfg["image_size"]),
        device=device,
        batch_size=int(args.fps_batch_size),
        warmup_iters=int(args.fps_warmup_iters),
        measure_iters=int(args.fps_measure_iters),
    )
    metrics = evaluate_coco_subset(
        model=model,
        dataset=dataset,
        device=device,
        metadata_path=Path(args.metadata).resolve(),
        manifest_path=Path(args.manifest).resolve(),
        image_size=int(data_cfg["image_size"]),
        num_classes=int(data_cfg["num_classes"]),
        num_boxes=num_boxes,
        anchors=anchors_by_level if anchors_by_level else anchors,
        box_parameterization=box_parameterization,
        max_samples=args.max_samples,
        score_threshold=float(args.score_threshold),
        score_alpha=float(args.score_alpha),
        score_beta=float(args.score_beta),
        top_k=int(args.top_k),
        nms_iou_threshold=float(args.nms_iou_threshold),
    )

    result = {
        "config_path": str(Path(args.config).resolve()),
        "checkpoint_path": str(Path(args.checkpoint).resolve()),
        "manifest_path": str(Path(args.manifest).resolve()),
        "device": str(device),
        "model_name": str(model_cfg["name"]),
        "num_boxes": num_boxes,
        "anchors": anchors_by_level if anchors_by_level else anchors,
        "box_parameterization": box_parameterization,
        "score_alpha": float(args.score_alpha),
        "score_beta": float(args.score_beta),
        "output_shape": output_shape,
        "params_total": params["total"],
        "params_trainable": params["trainable"],
        **flops,
        **fps,
        **metrics,
    }

    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output_json:
        output_path = Path(args.output_json).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
