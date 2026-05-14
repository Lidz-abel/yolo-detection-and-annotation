"""Benchmark PyTorch, TorchScript, and ONNX predictors with shared post-processing."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from backend.onnx_predictor import ONNXPredictor
from backend.pytorch_predictor import PyTorchPredictor
from backend.torchscript_predictor import TorchScriptPredictor
from data.detection_dataset import DetectionDataset
from tools.evaluate_coco import build_dataset_storage_kwargs
from utils.config import load_config, parse_anchor_map, parse_anchor_string, parse_int_list, parse_string_list
from utils.visualization import tensor_to_pil


DEFAULT_CONFIG = "configs/dual_scale_three_box_coco_only_noobj1_416_basic_aug_scale_jitter_50_lr7e4.toml"
DEFAULT_CHECKPOINT = (
    "outputs/dual_scale_three_box_coco_only_noobj1_416_basic_aug_scale_jitter_50_lr7e4_ddp_20260512_130823/best.pth"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark exported YOLO inference paths.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--torchscript-model", default="exports/checkpoint8/best_yolofinal_416.torchscript.pt")
    parser.add_argument("--onnx-model", default="exports/checkpoint8/best_yolofinal_416.onnx")
    parser.add_argument("--metadata", default="/home/lidz/YOLO/DataSet/Unified/metadata/class_maps.json")
    parser.add_argument("--manifest", default="/home/lidz/YOLO/DataSet/Unified/manifests/coco2017_val.jsonl")
    parser.add_argument("--formats", default="pytorch,torchscript,onnx")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--vis-samples", type=int, default=16)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--nms-iou-threshold", type=float, default=0.5)
    parser.add_argument("--output-json", default="outputs/export_benchmark/checkpoint8_benchmark.json")
    parser.add_argument("--vis-dir", default="outputs/export_benchmark/vis16")
    return parser.parse_args()


def build_dataset(config: dict, manifest_path: str, max_samples: int):
    data_cfg = config["data"]
    model_cfg = config["model"]
    grid_sizes = parse_int_list(data_cfg.get("grid_sizes"))
    if not grid_sizes:
        grid_sizes = [int(data_cfg["grid_size"])]
    feature_levels = parse_string_list(model_cfg.get("feature_levels"))
    if not feature_levels:
        feature_levels = [f"scale_{index}" for index in range(len(grid_sizes))]
    return DetectionDataset(
        manifest_path=manifest_path,
        image_size=int(data_cfg["image_size"]),
        grid_size=int(data_cfg.get("grid_size", grid_sizes[0])),
        grid_sizes=grid_sizes,
        feature_levels=feature_levels,
        num_classes=int(data_cfg["num_classes"]),
        num_boxes=int(model_cfg.get("num_boxes", 1)),
        anchors=parse_anchor_string(model_cfg.get("anchors")),
        anchors_by_level=parse_anchor_map(model_cfg, feature_levels),
        anchor_positive_iou=float(model_cfg.get("anchor_positive_iou", 0.25)),
        anchor_ignore_iou=float(model_cfg.get("anchor_ignore_iou", 0.5)),
        anchor_match_metric=str(model_cfg.get("anchor_match_metric", "iou")),
        anchor_shape_ratio=float(model_cfg.get("anchor_shape_ratio", 4.0)),
        anchor_ignore_shape_ratio=model_cfg.get("anchor_ignore_shape_ratio"),
        max_samples=max_samples,
        **build_dataset_storage_kwargs(data_cfg),
    )


def build_predictors(args):
    requested = [item.strip().lower() for item in args.formats.split(",") if item.strip()]
    predictors = {}
    for name in requested:
        try:
            if name == "pytorch":
                predictors[name] = PyTorchPredictor(
                    config_path=Path(args.config),
                    checkpoint_path=Path(args.checkpoint),
                    device_name=args.device,
                    metadata_path=Path(args.metadata),
                    use_fp16=False,
                )
            elif name == "torchscript":
                predictors[name] = TorchScriptPredictor(
                    model_path=Path(args.torchscript_model),
                    config_path=Path(args.config),
                    device_name=args.device,
                    metadata_path=Path(args.metadata),
                    use_fp16=False,
                )
            elif name == "onnx":
                predictors[name] = ONNXPredictor(
                    model_path=Path(args.onnx_model),
                    config_path=Path(args.config),
                    metadata_path=Path(args.metadata),
                )
            else:
                raise ValueError(f"unknown format: {name}")
        except Exception as exc:
            predictors[name] = exc
    return predictors


def percentile(values: list[float], point: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(max(round(point * (len(ordered) - 1)), 0), len(ordered) - 1)
    return float(ordered[index])


def summarize_latencies(latencies: list[dict]) -> dict:
    totals = [item["total"] for item in latencies]
    inference = [item["inference"] for item in latencies]
    return {
        "samples": len(latencies),
        "total_ms_mean": float(statistics.mean(totals)) if totals else 0.0,
        "total_ms_p50": percentile(totals, 0.50),
        "total_ms_p95": percentile(totals, 0.95),
        "inference_ms_mean": float(statistics.mean(inference)) if inference else 0.0,
        "fps_end_to_end": 1000.0 / max(float(statistics.mean(totals)) if totals else 0.0, 1e-9),
    }


def draw_predictions(image, bboxes: list[dict]):
    from PIL import ImageDraw

    drawn = image.copy()
    draw = ImageDraw.Draw(drawn)
    for box in bboxes:
        x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
        label = f"{box.get('class_name', box['class_id'])} {box['score']:.2f}"
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1 + 2, max(y1 - 12, 0)), label, fill="red")
    return drawn


def main():
    args = parse_args()
    config = load_config(Path(args.config).resolve())
    dataset = build_dataset(config, args.manifest, max(int(args.num_samples), int(args.vis_samples)))
    images = []
    sample_ids = []
    for index in range(min(len(dataset), max(int(args.num_samples), int(args.vis_samples)))):
        image_tensor, target = dataset[index]
        images.append(tensor_to_pil(image_tensor))
        sample_ids.append(str(target["sample_id"]))

    predictors = build_predictors(args)
    results = {
        "config_path": str(Path(args.config).resolve()),
        "checkpoint_path": str(Path(args.checkpoint).resolve()),
        "torchscript_model": str(Path(args.torchscript_model).resolve()),
        "onnx_model": str(Path(args.onnx_model).resolve()),
        "score_threshold": float(args.score_threshold),
        "top_k": int(args.top_k),
        "nms_iou_threshold": float(args.nms_iou_threshold),
        "formats": {},
    }
    vis_dir = Path(args.vis_dir).resolve()
    vis_dir.mkdir(parents=True, exist_ok=True)

    for name, predictor in predictors.items():
        if isinstance(predictor, Exception):
            results["formats"][name] = {"available": False, "error": str(predictor)}
            continue

        for image in images[: int(args.warmup)]:
            predictor.predict(image, args.score_threshold, args.top_k, args.nms_iou_threshold)
        latencies = []
        predictions_json = []
        start = time.perf_counter()
        for sample_id, image in zip(sample_ids[: int(args.num_samples)], images[: int(args.num_samples)]):
            result = predictor.predict(image, args.score_threshold, args.top_k, args.nms_iou_threshold)
            latencies.append(result["latency_ms"])
            predictions_json.append({"sample_id": sample_id, "bboxes": result["bboxes"], "latency_ms": result["latency_ms"]})
        wall_time = time.perf_counter() - start
        summary = summarize_latencies(latencies)
        summary["wall_time_seconds"] = wall_time
        summary["wall_time_fps"] = len(latencies) / max(wall_time, 1e-9)
        summary["first_sample_latency_ms"] = predictions_json[0]["latency_ms"] if predictions_json else {}
        results["formats"][name] = {
            "available": True,
            "summary": summary,
            "predictions": predictions_json,
        }

        format_vis_dir = vis_dir / name
        format_vis_dir.mkdir(parents=True, exist_ok=True)
        saved = []
        for item, image in zip(predictions_json[: int(args.vis_samples)], images[: int(args.vis_samples)]):
            output_path = format_vis_dir / f"{item['sample_id']}.png"
            draw_predictions(image, item["bboxes"]).save(output_path)
            saved.append(str(output_path))
        results["formats"][name]["visualizations"] = saved

    output_path = Path(args.output_json).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
