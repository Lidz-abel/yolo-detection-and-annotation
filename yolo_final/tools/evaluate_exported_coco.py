"""Run COCO evaluation for exported TorchScript detector artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from data.detection_dataset import DetectionDataset
from tools.evaluate_coco import build_dataset_storage_kwargs
from utils.coco_eval import evaluate_coco_subset
from utils.config import load_config, parse_anchor_map, parse_anchor_string, parse_int_list, parse_string_list


DEFAULT_CONFIG = "configs/dual_scale_three_box_coco_only_noobj1_416_basic_aug_scale_jitter_50_lr7e4.toml"
DEFAULT_MANIFEST = "/home/lidz/YOLO/DataSet/Unified/manifests/coco2017_val.jsonl"
DEFAULT_METADATA = "/home/lidz/YOLO/DataSet/Unified/metadata/class_maps.json"
DEFAULT_TORCHSCRIPT = "exports/checkpoint8/best_yolofinal_416_lr7e4_int8_backbone_calib128.torchscript.pt"


class TorchScriptTupleToDict(nn.Module):
    """Wrap exported tuple outputs back into the dict expected by COCO eval."""

    def __init__(self, model_path: Path, feature_levels: list[str], device: torch.device):
        super().__init__()
        self.model_path = Path(model_path).resolve()
        self.feature_levels = list(feature_levels)
        if not self.model_path.exists():
            raise FileNotFoundError(f"TorchScript model not found: {self.model_path}")
        self.model = torch.jit.load(str(self.model_path), map_location=device).eval()

    def forward(self, images: torch.Tensor):
        outputs = self.model(images)
        if isinstance(outputs, dict):
            return outputs
        if isinstance(outputs, torch.Tensor):
            return {self.feature_levels[0]: outputs}
        return {
            level: outputs[index]
            for index, level in enumerate(self.feature_levels)
        }


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate an exported TorchScript detector on COCO.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--torchscript-model", default=DEFAULT_TORCHSCRIPT)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--metadata", default=DEFAULT_METADATA)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--score-threshold", type=float, default=None)
    parser.add_argument("--score-alpha", type=float, default=None)
    parser.add_argument("--score-beta", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--nms-iou-threshold", type=float, default=None)
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def build_device(device_name: str) -> torch.device:
    if str(device_name).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def build_dataset(config: dict, manifest_path: str, max_samples: int) -> tuple[DetectionDataset, list[str], list[int]]:
    data_cfg = config["data"]
    model_cfg = config["model"]
    grid_sizes = parse_int_list(data_cfg.get("grid_sizes"))
    if not grid_sizes:
        grid_sizes = [int(data_cfg["grid_size"])]
    feature_levels = parse_string_list(model_cfg.get("feature_levels"))
    if not feature_levels:
        feature_levels = [f"scale_{index}" for index in range(len(grid_sizes))]
    dataset = DetectionDataset(
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
    return dataset, feature_levels, grid_sizes


def main():
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    data_cfg = config["data"]
    model_cfg = config["model"]
    eval_cfg = config.get("evaluation", {})
    dataset, feature_levels, _grid_sizes = build_dataset(config, args.manifest, int(args.max_samples))
    device = build_device(args.device)
    model = TorchScriptTupleToDict(Path(args.torchscript_model), feature_levels, device).to(device).eval()
    anchors = parse_anchor_string(model_cfg.get("anchors"))
    anchors_by_level = parse_anchor_map(model_cfg, feature_levels)

    score_threshold = float(args.score_threshold if args.score_threshold is not None else eval_cfg.get("score_threshold", 0.05))
    score_alpha = float(args.score_alpha if args.score_alpha is not None else eval_cfg.get("score_alpha", 1.0))
    score_beta = float(args.score_beta if args.score_beta is not None else eval_cfg.get("score_beta", 1.0))
    top_k = int(args.top_k if args.top_k is not None else eval_cfg.get("top_k", 100))
    nms_iou_threshold = float(
        args.nms_iou_threshold if args.nms_iou_threshold is not None else eval_cfg.get("nms_iou_threshold", 0.5)
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
        anchors=anchors_by_level if anchors_by_level else anchors,
        box_parameterization=str(model_cfg.get("box_parameterization", "legacy")),
        max_samples=int(args.max_samples),
        score_threshold=score_threshold,
        score_alpha=score_alpha,
        score_beta=score_beta,
        top_k=top_k,
        nms_iou_threshold=nms_iou_threshold,
    )
    result = {
        "format": "torchscript",
        "config_path": str(config_path),
        "torchscript_model": str(Path(args.torchscript_model).resolve()),
        "manifest_path": str(Path(args.manifest).resolve()),
        "metadata_path": str(Path(args.metadata).resolve()),
        "device": str(device),
        "feature_levels": feature_levels,
        "num_boxes": int(model_cfg.get("num_boxes", 1)),
        "box_parameterization": str(model_cfg.get("box_parameterization", "legacy")),
        "score_threshold": score_threshold,
        "score_alpha": score_alpha,
        "score_beta": score_beta,
        "top_k": top_k,
        "nms_iou_threshold": nms_iou_threshold,
        **metrics,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output_json:
        output_path = Path(args.output_json).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
