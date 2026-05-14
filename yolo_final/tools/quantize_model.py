"""Post-training static INT8 quantization for the YOLO detector."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from data.detection_dataset import DetectionDataset
from models.detector import YOLOv0Baseline
from tools.evaluate_coco import build_dataset_storage_kwargs
from tools.export_model import TupleOutputWrapper
from utils.config import (
    load_config,
    parse_anchor_map,
    parse_anchor_string,
    parse_int_list,
    parse_string_list,
)


DEFAULT_CONFIG = "configs/dual_scale_three_box_coco_only_noobj1_416_basic_aug_scale_jitter_50_lr7e4.toml"
DEFAULT_CHECKPOINT = (
    "outputs/dual_scale_three_box_coco_only_noobj1_416_basic_aug_scale_jitter_50_lr7e4_ddp_20260512_130823/best.pth"
)
DEFAULT_MANIFEST = "/home/lidz/YOLO/DataSet/Unified/manifests/coco2017_val.jsonl"


def parse_args():
    parser = argparse.ArgumentParser(description="Quantize YOLO detector with FX PTQ.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", default="exports/checkpoint8")
    parser.add_argument("--prefix", default="best_yolofinal_416_lr7e4_int8")
    parser.add_argument("--calibration-samples", type=int, default=128)
    parser.add_argument("--backend", default="x86", choices=["x86", "fbgemm", "qnnpack", "onednn"])
    parser.add_argument(
        "--float-module-regex",
        default=".*head.*",
        help="Regex for modules kept in FP32. Default keeps the detection head unquantized for PTQ stability.",
    )
    parser.add_argument("--strict-trace", action="store_true")
    return parser.parse_args()


def build_model(config: dict, checkpoint_path: Path) -> nn.Module:
    data_cfg = config["data"]
    model_cfg = config["model"]
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
    )
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model


def build_calibration_dataset(config: dict, manifest_path: str, max_samples: int) -> DetectionDataset:
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


def calibration_batches(dataset: DetectionDataset, count: int):
    limit = min(len(dataset), int(count))
    for index in range(limit):
        image, _target = dataset[index]
        yield image.unsqueeze(0)


def max_abs_diff(reference, candidate) -> dict[str, float]:
    result = {}
    for index, (ref, got) in enumerate(zip(reference, candidate)):
        diff = (ref.detach().cpu() - got.detach().cpu()).abs()
        result[f"output_{index}_max_abs_diff"] = float(diff.max().item())
        result[f"output_{index}_mean_abs_diff"] = float(diff.mean().item())
    return result


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(Path(args.config).resolve())
    feature_levels = parse_string_list(config["model"].get("feature_levels")) or ["output"]
    image_size = int(config["data"]["image_size"])
    torch.backends.quantized.engine = args.backend

    float_model = build_model(config, Path(args.checkpoint).resolve())
    float_wrapper = TupleOutputWrapper(float_model, feature_levels).eval()
    example_input = torch.randn(1, 3, image_size, image_size)

    qconfig_mapping = get_default_qconfig_mapping(args.backend)
    if args.float_module_regex:
        qconfig_mapping = qconfig_mapping.set_module_name_regex(str(args.float_module_regex), None)
    prepare_start = time.perf_counter()
    prepared = prepare_fx(float_wrapper, qconfig_mapping, example_inputs=(example_input,))
    prepare_seconds = time.perf_counter() - prepare_start

    dataset = build_calibration_dataset(config, args.manifest, int(args.calibration_samples))
    calibrate_start = time.perf_counter()
    with torch.inference_mode():
        for batch in calibration_batches(dataset, int(args.calibration_samples)):
            prepared(batch)
    calibrate_seconds = time.perf_counter() - calibrate_start

    convert_start = time.perf_counter()
    quantized = convert_fx(prepared).eval()
    convert_seconds = time.perf_counter() - convert_start

    with torch.inference_mode():
        reference_outputs = float_wrapper(example_input)
        quantized_outputs = quantized(example_input)
        correctness = max_abs_diff(reference_outputs, quantized_outputs)

    script_path = output_dir / f"{args.prefix}.torchscript.pt"
    trace_start = time.perf_counter()
    with torch.inference_mode():
        traced = torch.jit.trace(quantized, example_input, strict=bool(args.strict_trace))
        traced.save(str(script_path))
    trace_seconds = time.perf_counter() - trace_start

    metadata = {
        "format": "torchscript_int8_fx_ptq",
        "config_path": str(Path(args.config).resolve()),
        "checkpoint_path": str(Path(args.checkpoint).resolve()),
        "manifest_path": str(Path(args.manifest).resolve()),
        "image_size": image_size,
        "feature_levels": feature_levels,
        "backend": args.backend,
        "float_module_regex": str(args.float_module_regex),
        "calibration_samples_requested": int(args.calibration_samples),
        "calibration_samples_used": min(len(dataset), int(args.calibration_samples)),
        "artifact": {
            "path": str(script_path),
            "bytes": script_path.stat().st_size,
        },
        "correctness_on_random_input": correctness,
        "timing_seconds": {
            "prepare_fx": prepare_seconds,
            "calibration": calibrate_seconds,
            "convert_fx": convert_seconds,
            "trace": trace_seconds,
        },
        "notes": [
            "This is CPU post-training static INT8 quantization with FX graph mode.",
            "The exported model keeps raw prediction tensors in feature-level tuple order.",
            "Use YOLO_BACKEND_MODEL_FORMAT=torchscript and YOLO_BACKEND_DEVICE=cpu for backend demos.",
            "Accuracy-sensitive conclusions require COCO eval or decoded prediction parity after quantization.",
        ],
    }
    metadata_path = output_dir / f"{args.prefix}.metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
