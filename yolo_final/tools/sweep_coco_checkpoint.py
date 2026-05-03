"""Run non-training COCO metric sweeps for one detector checkpoint."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from data.detection_dataset import DetectionDataset
from models.detector import YOLOv0Baseline
from tools.evaluate import build_device
from tools.evaluate_coco import build_dataset_storage_kwargs, load_state_dict
from utils.coco_eval import evaluate_coco_subset
from utils.config import load_config, parse_anchor_map, parse_anchor_string, parse_int_list, parse_string_list
from utils.modeling import count_parameters, describe_model_output


def parse_float_list(raw: str) -> list[float]:
    """Parse comma-separated floats."""
    return [float(item.strip()) for item in str(raw).split(",") if item.strip()]


def parse_int_list_arg(raw: str) -> list[int]:
    """Parse comma-separated ints."""
    return [int(item.strip()) for item in str(raw).split(",") if item.strip()]


def parse_args():
    """Parse sweep inputs."""
    parser = argparse.ArgumentParser(description="COCO non-training sweep for one checkpoint.")
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
    parser.add_argument("--score-alphas", type=str, default="1.0,1.5,2.0", help="Comma-separated score alphas.")
    parser.add_argument("--score-betas", type=str, default="1.0", help="Comma-separated score betas.")
    parser.add_argument(
        "--score-thresholds",
        type=str,
        default="0.01,0.03,0.05,0.1",
        help="Comma-separated score thresholds.",
    )
    parser.add_argument("--top-ks", type=str, default="50,100,200", help="Comma-separated top-k values.")
    parser.add_argument(
        "--nms-iou-thresholds",
        type=str,
        default="0.4,0.5,0.6",
        help="Comma-separated class-wise NMS IoU thresholds.",
    )
    parser.add_argument("--output-json", type=str, required=True, help="JSON output path.")
    return parser.parse_args()


def build_dataset(config: dict, manifest_path: str, max_samples: int):
    """Build a packed COCO dataset for evaluation."""
    data_cfg = config["data"]
    model_cfg = config["model"]
    anchors = parse_anchor_string(model_cfg.get("anchors"))
    grid_sizes = parse_int_list(data_cfg.get("grid_sizes"))
    if not grid_sizes:
        grid_sizes = [int(data_cfg["grid_size"])]
    feature_levels = parse_string_list(model_cfg.get("feature_levels"))
    if not feature_levels:
        feature_levels = [f"scale_{index}" for index in range(len(grid_sizes))]
    default_grid_size = int(data_cfg.get("grid_size", grid_sizes[0]))
    return DetectionDataset(
        manifest_path=manifest_path,
        image_size=int(data_cfg["image_size"]),
        grid_size=default_grid_size,
        grid_sizes=grid_sizes,
        feature_levels=feature_levels,
        num_classes=int(data_cfg["num_classes"]),
        num_boxes=int(model_cfg.get("num_boxes", 1)),
        anchors=anchors,
        anchors_by_level=parse_anchor_map(model_cfg, feature_levels),
        anchor_positive_iou=float(model_cfg.get("anchor_positive_iou", 0.25)),
        anchor_ignore_iou=float(model_cfg.get("anchor_ignore_iou", 0.5)),
        anchor_match_metric=str(model_cfg.get("anchor_match_metric", "iou")),
        anchor_shape_ratio=float(model_cfg.get("anchor_shape_ratio", 4.0)),
        anchor_ignore_shape_ratio=model_cfg.get("anchor_ignore_shape_ratio"),
        max_samples=max_samples,
        **build_dataset_storage_kwargs(data_cfg),
    )


def build_model(config: dict, device: torch.device):
    """Build a detector matching the training config."""
    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["train"]
    feature_levels = parse_string_list(model_cfg.get("feature_levels"))
    model = YOLOv0Baseline(
        num_classes=int(data_cfg["num_classes"]),
        model_name=str(model_cfg["name"]),
        width_mult=float(model_cfg["width_mult"]),
        depth_mult=float(model_cfg["depth_mult"]),
        use_residual=bool(model_cfg["use_residual"]),
        num_boxes=int(model_cfg.get("num_boxes", 1)),
        head_type=str(model_cfg.get("head_type", "shared")),
        neck_type=str(model_cfg.get("neck_type", "none")),
        feature_levels=feature_levels,
    ).to(device)
    if device.type == "cuda" and torch.cuda.device_count() > 1 and bool(train_cfg["use_data_parallel"]):
        model = torch.nn.DataParallel(model)
    return model


def sort_top(results: list[dict], key: str, limit: int = 10) -> list[dict]:
    """Return compact top configs sorted by one metric."""
    return sorted(results, key=lambda item: item.get(key, 0.0), reverse=True)[:limit]


def main():
    """Run the configured sweep and write JSON results."""
    args = parse_args()
    config = load_config(Path(args.config).resolve())
    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["train"]
    device = build_device(str(train_cfg["device"]))
    feature_levels = parse_string_list(model_cfg.get("feature_levels"))
    anchors = parse_anchor_map(model_cfg, feature_levels) or parse_anchor_string(model_cfg.get("anchors"))
    box_parameterization = str(model_cfg.get("box_parameterization", "legacy"))

    dataset = build_dataset(config, args.manifest, args.max_samples)
    model = build_model(config, device)
    checkpoint = torch.load(Path(args.checkpoint).resolve(), map_location=device)
    load_state_dict(model, checkpoint)
    model.eval()

    output_shape = describe_model_output(model, int(data_cfg["image_size"]), device)
    params = count_parameters(model)

    score_alphas = parse_float_list(args.score_alphas)
    score_betas = parse_float_list(args.score_betas)
    score_thresholds = parse_float_list(args.score_thresholds)
    top_ks = parse_int_list_arg(args.top_ks)
    nms_iou_thresholds = parse_float_list(args.nms_iou_thresholds)

    results = []
    combos = list(itertools.product(score_alphas, score_betas, score_thresholds, top_ks, nms_iou_thresholds))
    for combo_index, (score_alpha, score_beta, score_threshold, top_k, nms_iou_threshold) in enumerate(combos, start=1):
        print(
            "sweep "
            f"{combo_index}/{len(combos)}: alpha={score_alpha}, beta={score_beta}, "
            f"threshold={score_threshold}, top_k={top_k}, nms={nms_iou_threshold}",
            flush=True,
        )
        metrics = evaluate_coco_subset(
            model=model,
            dataset=dataset,
            device=device,
            metadata_path=Path(args.metadata).resolve(),
            manifest_path=Path(args.manifest).resolve(),
            image_size=int(data_cfg["image_size"]),
            num_classes=int(data_cfg["num_classes"]),
            num_boxes=int(model_cfg.get("num_boxes", 1)),
            anchors=anchors,
            box_parameterization=box_parameterization,
            max_samples=int(args.max_samples),
            score_threshold=float(score_threshold),
            top_k=int(top_k),
            nms_iou_threshold=float(nms_iou_threshold),
            score_alpha=float(score_alpha),
            score_beta=float(score_beta),
        )
        results.append(
            {
                "score_alpha": float(score_alpha),
                "score_beta": float(score_beta),
                "score_threshold": float(score_threshold),
                "top_k": int(top_k),
                "nms_iou_threshold": float(nms_iou_threshold),
                **metrics,
            }
        )

    result = {
        "config_path": str(Path(args.config).resolve()),
        "checkpoint_path": str(Path(args.checkpoint).resolve()),
        "manifest_path": str(Path(args.manifest).resolve()),
        "metadata_path": str(Path(args.metadata).resolve()),
        "max_samples": int(args.max_samples),
        "num_combinations": len(results),
        "device": str(device),
        "model_name": str(model_cfg["name"]),
        "num_boxes": int(model_cfg.get("num_boxes", 1)),
        "anchors": anchors,
        "box_parameterization": box_parameterization,
        "output_shape": output_shape,
        "params_total": params["total"],
        "params_trainable": params["trainable"],
        "sweep_space": {
            "score_alphas": score_alphas,
            "score_betas": score_betas,
            "score_thresholds": score_thresholds,
            "top_ks": top_ks,
            "nms_iou_thresholds": nms_iou_thresholds,
        },
        "top_by_ap": sort_top(results, "coco_ap"),
        "top_by_ap50": sort_top(results, "coco_ap50"),
        "top_by_ar100": sort_top(results, "coco_ar100"),
        "results": results,
    }
    output_path = Path(args.output_json).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
