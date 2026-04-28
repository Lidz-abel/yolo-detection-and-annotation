"""Run non-training COCO diagnostics for one detector checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
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
from utils.prediction import decode_predictions_for_image, select_prediction_for_image
from utils.visualization import draw_gt_and_predictions, tensor_to_pil


def parse_args():
    """Parse diagnostic inputs."""
    parser = argparse.ArgumentParser(description="Non-training COCO checkpoint diagnostics.")
    parser.add_argument("--config", type=str, required=True, help="Config used to build the model and dataset.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint to diagnose.")
    parser.add_argument(
        "--manifest",
        type=str,
        default="/home/lidz/YOLO/DataSet/Unified/manifests/coco2017_val.jsonl",
        help="COCO manifest to evaluate.",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default="/home/lidz/YOLO/DataSet/Unified/metadata/class_maps.json",
        help="COCO metadata with category mappings.",
    )
    parser.add_argument("--max-samples", type=int, default=1000, help="Number of samples for diagnostic sweeps.")
    parser.add_argument("--vis-samples", type=int, default=8, help="Number of visualization samples.")
    parser.add_argument("--output-json", type=str, required=True, help="Diagnostic JSON output path.")
    parser.add_argument("--vis-dir", type=str, required=True, help="Visualization output directory.")
    return parser.parse_args()


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


def build_dataset(config: dict, manifest_path: str, max_samples: int):
    """Build a packed COCO dataset for diagnostics."""
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


def quantiles(values: list[float], points: list[float]) -> dict[str, float]:
    """Return simple quantiles for one numeric list."""
    if not values:
        return {str(point): 0.0 for point in points}
    ordered = sorted(values)
    result = {}
    for point in points:
        index = min(max(round(point * (len(ordered) - 1)), 0), len(ordered) - 1)
        result[str(point)] = float(ordered[index])
    return result


def collect_prediction_stats(
    model,
    dataset,
    device,
    config: dict,
    anchors,
    max_samples: int,
) -> dict:
    """Collect score and class distribution stats without changing weights."""
    data_cfg = config["data"]
    model_cfg = config["model"]
    model.eval()
    scores = []
    per_image_counts = []
    class_counts = Counter()
    with torch.no_grad():
        for sample_index in range(min(max_samples, len(dataset))):
            image_tensor, _target = dataset[sample_index]
            pred = select_prediction_for_image(model(image_tensor.unsqueeze(0).to(device)), 0)
            predictions = decode_predictions_for_image(
                pred=pred,
                image_size=int(data_cfg["image_size"]),
                num_classes=int(data_cfg["num_classes"]),
                num_boxes=int(model_cfg.get("num_boxes", 1)),
                anchors=anchors,
                box_parameterization=str(model_cfg.get("box_parameterization", "legacy")),
                score_threshold=0.0,
                top_k=100,
                nms_iou_threshold=0.5,
                score_alpha=2.0,
                score_beta=1.0,
            )
            per_image_counts.append(len(predictions))
            for prediction in predictions:
                scores.append(float(prediction["score"]))
                class_counts[int(prediction["class_id"])] += 1
    top_classes = [
        {"class_id": class_id, "count": count}
        for class_id, count in class_counts.most_common(15)
    ]
    return {
        "samples": min(max_samples, len(dataset)),
        "prediction_count": len(scores),
        "predictions_per_image": {
            "mean": float(sum(per_image_counts) / max(len(per_image_counts), 1)),
            "min": int(min(per_image_counts) if per_image_counts else 0),
            "max": int(max(per_image_counts) if per_image_counts else 0),
        },
        "score_quantiles": quantiles(scores, [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]),
        "top_predicted_classes": top_classes,
    }


def run_metric_sweeps(model, dataset, device, config, metadata_path: Path, manifest_path: Path, anchors, max_samples: int):
    """Run focused score, top-k, and NMS sweeps."""
    data_cfg = config["data"]
    model_cfg = config["model"]
    common = {
        "model": model,
        "dataset": dataset,
        "device": device,
        "metadata_path": metadata_path,
        "manifest_path": manifest_path,
        "image_size": int(data_cfg["image_size"]),
        "num_classes": int(data_cfg["num_classes"]),
        "num_boxes": int(model_cfg.get("num_boxes", 1)),
        "anchors": anchors,
        "box_parameterization": str(model_cfg.get("box_parameterization", "legacy")),
        "max_samples": max_samples,
        "score_alpha": 2.0,
        "score_beta": 1.0,
    }
    sweeps = {"score_threshold": [], "top_k": [], "nms_iou_threshold": []}
    for score_threshold in [0.01, 0.03, 0.05, 0.1, 0.2, 0.3]:
        metrics = evaluate_coco_subset(
            **common,
            score_threshold=score_threshold,
            top_k=100,
            nms_iou_threshold=0.5,
        )
        sweeps["score_threshold"].append({"value": score_threshold, **metrics})
    for top_k in [20, 50, 100, 200]:
        metrics = evaluate_coco_subset(
            **common,
            score_threshold=0.05,
            top_k=top_k,
            nms_iou_threshold=0.5,
        )
        sweeps["top_k"].append({"value": top_k, **metrics})
    for nms_iou_threshold in [0.4, 0.5, 0.6, 0.7]:
        metrics = evaluate_coco_subset(
            **common,
            score_threshold=0.05,
            top_k=100,
            nms_iou_threshold=nms_iou_threshold,
        )
        sweeps["nms_iou_threshold"].append({"value": nms_iou_threshold, **metrics})
    return sweeps


def save_visualizations(model, dataset, device, config, anchors, output_dir: Path, max_samples: int):
    """Write COCO GT-vs-pred images for qualitative inspection."""
    data_cfg = config["data"]
    model_cfg = config["model"]
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    model.eval()
    with torch.no_grad():
        for sample_index in range(min(max_samples, len(dataset))):
            image_tensor, target = dataset[sample_index]
            pred = select_prediction_for_image(model(image_tensor.unsqueeze(0).to(device)), 0)
            predictions = decode_predictions_for_image(
                pred=pred,
                image_size=int(data_cfg["image_size"]),
                num_classes=int(data_cfg["num_classes"]),
                num_boxes=int(model_cfg.get("num_boxes", 1)),
                anchors=anchors,
                box_parameterization=str(model_cfg.get("box_parameterization", "legacy")),
                score_threshold=0.05,
                top_k=20,
                nms_iou_threshold=0.5,
                score_alpha=2.0,
                score_beta=1.0,
            )
            image = tensor_to_pil(image_tensor)
            image = draw_gt_and_predictions(image, target["boxes"], target["labels"], predictions)
            output_path = output_dir / f"{target['sample_id']}.png"
            image.save(output_path)
            saved.append(str(output_path))
    return saved


def main():
    """Run all diagnostics and write artifacts."""
    args = parse_args()
    config = load_config(Path(args.config).resolve())
    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["train"]
    device = build_device(str(train_cfg["device"]))
    feature_levels = parse_string_list(model_cfg.get("feature_levels"))
    anchors = parse_anchor_map(model_cfg, feature_levels) or parse_anchor_string(model_cfg.get("anchors"))

    dataset = build_dataset(config, args.manifest, args.max_samples)
    model = build_model(config, device)
    checkpoint = torch.load(Path(args.checkpoint).resolve(), map_location=device)
    load_state_dict(model, checkpoint)

    sweeps = run_metric_sweeps(
        model=model,
        dataset=dataset,
        device=device,
        config=config,
        metadata_path=Path(args.metadata).resolve(),
        manifest_path=Path(args.manifest).resolve(),
        anchors=anchors,
        max_samples=min(args.max_samples, len(dataset)),
    )
    stats = collect_prediction_stats(
        model=model,
        dataset=dataset,
        device=device,
        config=config,
        anchors=anchors,
        max_samples=min(args.max_samples, len(dataset)),
    )
    visualizations = save_visualizations(
        model=model,
        dataset=dataset,
        device=device,
        config=config,
        anchors=anchors,
        output_dir=Path(args.vis_dir).resolve(),
        max_samples=args.vis_samples,
    )

    result = {
        "config_path": str(Path(args.config).resolve()),
        "checkpoint_path": str(Path(args.checkpoint).resolve()),
        "manifest_path": str(Path(args.manifest).resolve()),
        "max_samples": int(min(args.max_samples, len(dataset))),
        "storage_mode": dataset.storage_mode,
        "packed_index": str(dataset.packed_index_path) if dataset.packed_index_path else "",
        "sweeps": sweeps,
        "prediction_stats": stats,
        "visualizations": visualizations,
    }
    output_path = Path(args.output_json).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
