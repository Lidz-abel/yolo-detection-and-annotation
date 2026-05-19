"""Verify saved desktop annotations by running one tiny training epoch.

This script is intentionally small and local. It checks the YOLO txt files,
draws ground-truth visualizations, loads the best trainable checkpoint, and
runs one epoch on the newly saved annotation images. The goal is format
verification, not meaningful model improvement.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import numpy as np
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from losses.yolo_loss import YOLOLoss  # noqa: E402
from models.detector import YOLOv0Baseline  # noqa: E402
from utils.config import (  # noqa: E402
    load_config,
    parse_anchor_map,
    parse_anchor_string,
    parse_float_list,
    parse_float_map,
    parse_int_list,
    parse_string_list,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "dual_scale_three_box_coco_only_noobj1_416_basic_aug_scale_jitter_50_lr7e4.toml"
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "dual_scale_three_box_coco_only_noobj1_416_basic_aug_scale_jitter_50_lr7e4_ddp_20260512_130823"
    / "best.pth"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-root", type=Path, default=PROJECT_ROOT / "desktop_app_annotations")
    parser.add_argument("--config", type=Path, default=Path(os.getenv("YOLO_BACKEND_CONFIG", DEFAULT_CONFIG)))
    parser.add_argument("--checkpoint", type=Path, default=Path(os.getenv("YOLO_BACKEND_CHECKPOINT", DEFAULT_CHECKPOINT)))
    parser.add_argument("--lr", type=float, default=1e-4)
    return parser.parse_args()


def yolo_to_xyxy(row: str, width: int, height: int) -> tuple[int, list[float]]:
    parts = row.strip().split()
    if len(parts) != 5:
        raise ValueError(f"YOLO label row must have 5 fields, got {len(parts)}: {row!r}")
    class_id = int(float(parts[0]))
    cx, cy, bw, bh = [float(value) for value in parts[1:]]
    for value in (cx, cy, bw, bh):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"YOLO normalized value out of [0,1]: {row!r}")
    x1 = (cx - bw / 2.0) * width
    y1 = (cy - bh / 2.0) * height
    x2 = (cx + bw / 2.0) * width
    y2 = (cy + bh / 2.0) * height
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid bbox size: {row!r}")
    return class_id, [max(0.0, x1), max(0.0, y1), min(float(width), x2), min(float(height), y2)]


def collect_samples(annotation_root: Path) -> list[dict]:
    image_dir = annotation_root / "images"
    label_dir = annotation_root / "labels"
    if not image_dir.exists() or not label_dir.exists():
        raise FileNotFoundError(f"Missing annotation images/labels under {annotation_root}")
    samples = []
    for label_path in sorted(label_dir.glob("*.txt")):
        image_path = None
        for suffix in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            candidate = image_dir / f"{label_path.stem}{suffix}"
            if candidate.exists():
                image_path = candidate
                break
        if image_path is None:
            continue
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            width, height = image.size
            rows = [line for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            labels = []
            boxes = []
            for row in rows:
                class_id, box = yolo_to_xyxy(row, width, height)
                labels.append(class_id)
                boxes.append(box)
        if boxes:
            samples.append({"image_path": image_path, "label_path": label_path, "boxes": boxes, "labels": labels, "width": width, "height": height})
    if not samples:
        raise ValueError("No valid annotated samples found. Save at least one image with one bbox first.")
    return samples


def draw_gt_visualizations(samples: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for sample in samples:
        image = Image.open(sample["image_path"]).convert("RGB")
        draw = ImageDraw.Draw(image)
        for class_id, box in zip(sample["labels"], sample["boxes"]):
            x1, y1, x2, y2 = box
            draw.rectangle((x1, y1, x2, y2), outline="#22c55e", width=max(2, image.width // 320))
            draw.text((x1 + 3, max(0, y1 - 12)), str(class_id), fill="#22c55e")
        image.save(output_dir / f"{sample['label_path'].stem}_gt.jpg")


def resolve_feature_levels(data_cfg: dict, model_cfg: dict) -> tuple[list[str], list[int]]:
    grid_sizes = parse_int_list(data_cfg.get("grid_sizes"))
    if not grid_sizes:
        grid_sizes = [int(data_cfg.get("grid_size", 13))]
    feature_levels = parse_string_list(model_cfg.get("feature_levels"))
    if not feature_levels:
        feature_levels = [f"scale_{index}" for index in range(len(grid_sizes))]
    return feature_levels, grid_sizes


def build_model_and_loss(config: dict, checkpoint_path: Path, device: torch.device):
    data_cfg = config["data"]
    model_cfg = config["model"]
    loss_cfg = config["loss"]
    feature_levels, _grid_sizes = resolve_feature_levels(data_cfg, model_cfg)
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
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Trainable checkpoint not found: {checkpoint_path}\n"
            "TorchScript can only infer; closed-loop training verification needs best.pth."
        )
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith("head.")
    criterion = YOLOLoss(
        num_classes=int(data_cfg["num_classes"]),
        lambda_box=float(loss_cfg["lambda_box"]),
        lambda_obj=float(loss_cfg.get("lambda_obj", 1.0)),
        lambda_noobj=float(loss_cfg.get("lambda_noobj", 0.5)),
        lambda_cls=float(loss_cfg["lambda_cls"]),
        num_boxes=int(model_cfg.get("num_boxes", 1)),
        anchors=parse_anchor_string(model_cfg.get("anchors")),
        anchors_by_level=parse_anchor_map(model_cfg, feature_levels),
        box_parameterization=str(model_cfg.get("box_parameterization", "legacy")),
        soft_objectness_target=str(loss_cfg.get("soft_objectness_target", "hard")),
        soft_objectness_min=float(loss_cfg.get("soft_objectness_min", 0.0)),
        soft_classification_target=str(loss_cfg.get("soft_classification_target", "hard")),
        cls_loss_mode=str(loss_cfg.get("cls_loss_mode", "bce")),
        varifocal_alpha=float(loss_cfg.get("varifocal_alpha", 0.75)),
        varifocal_gamma=float(loss_cfg.get("varifocal_gamma", 2.0)),
        assignment_strategy=str(loss_cfg.get("assignment_strategy", "static")),
        dynamic_topk=int(loss_cfg.get("dynamic_topk", 2)),
        dynamic_center_radius=int(loss_cfg.get("dynamic_center_radius", 1)),
        dynamic_box_cost=float(loss_cfg.get("dynamic_box_cost", 3.0)),
        dynamic_cls_cost=float(loss_cfg.get("dynamic_cls_cost", 1.0)),
        dynamic_ignore_iou=float(loss_cfg.get("dynamic_ignore_iou", 0.5)),
        dynamic_anchor_shape_cost=float(loss_cfg.get("dynamic_anchor_shape_cost", 0.0)),
        scale_assignment=str(loss_cfg.get("scale_assignment", "all")),
        scale_area_threshold=float(loss_cfg.get("scale_area_threshold", 0.2)),
        scale_area_thresholds=parse_float_list(loss_cfg.get("scale_area_thresholds")),
        scale_loss_weights=parse_float_map(loss_cfg.get("scale_loss_weights")),
        feature_levels=feature_levels,
    )
    return model, criterion, feature_levels, _grid_sizes


def build_batch(sample: dict, config: dict, feature_levels: list[str], grid_sizes: list[int], device: torch.device):
    image_size = int(config["data"]["image_size"])
    num_boxes = int(config["model"].get("num_boxes", 1))
    num_classes = int(config["data"]["num_classes"])
    image = Image.open(sample["image_path"]).convert("RGB").resize((image_size, image_size), Image.BILINEAR)
    image_array = np.asarray(image, dtype=np.float32) / 255.0
    image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0)

    scale_x = image_size / float(sample["width"])
    scale_y = image_size / float(sample["height"])
    boxes = torch.tensor(
        [[x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y] for x1, y1, x2, y2 in sample["boxes"]],
        dtype=torch.float32,
    )
    labels = torch.tensor(sample["labels"], dtype=torch.long)
    multiscale_targets = {}
    for level, grid_size in zip(feature_levels, grid_sizes):
        multiscale_targets[level] = {
            "target_box": torch.zeros(1, grid_size, grid_size, num_boxes, 4),
            "target_obj": torch.zeros(1, grid_size, grid_size, num_boxes),
            "target_cls": torch.zeros(1, grid_size, grid_size, num_boxes, num_classes),
            "object_mask": torch.zeros(1, grid_size, grid_size, num_boxes),
            "noobj_mask": torch.ones(1, grid_size, grid_size, num_boxes),
            "collision_count": torch.zeros(1),
            "ignored_count": torch.zeros(1),
            "dropped_gt_count": torch.zeros(1),
        }
    targets = {
        "boxes": [boxes.to(device)],
        "labels": [labels.to(device)],
        "resized_size": torch.tensor([[image_size, image_size]], dtype=torch.float32, device=device),
        "multiscale_targets": {k: {kk: vv.to(device) for kk, vv in v.items()} for k, v in multiscale_targets.items()},
    }
    return image_tensor.to(device), targets


def evaluate_loss(model, criterion, samples, config, feature_levels, grid_sizes, device) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for sample in samples:
            image, targets = build_batch(sample, config, feature_levels, grid_sizes, device)
            total += float(criterion(model(image), targets)["total_loss"].item())
    return total / len(samples)


def save_loss_curve(losses: list[float], output_path: Path) -> None:
    width, height = 720, 320
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((50, 30, width - 30, height - 45), outline="#94a3b8")
    min_loss = min(losses)
    max_loss = max(losses)
    span = max(max_loss - min_loss, 1e-6)
    points = []
    for index, loss in enumerate(losses):
        x = 50 + index * (width - 80) / max(len(losses) - 1, 1)
        y = height - 45 - (loss - min_loss) / span * (height - 80)
        points.append((x, y))
    if len(points) > 1:
        draw.line(points, fill="#2563eb", width=3)
    for point in points:
        draw.ellipse((point[0] - 4, point[1] - 4, point[0] + 4, point[1] + 4), fill="#2563eb")
    draw.text((50, 8), "Closed-loop 1 epoch loss check", fill="#0f172a")
    draw.text((50, height - 32), f"initial={losses[0]:.4f} final={losses[-1]:.4f}", fill="#0f172a")
    image.save(output_path)


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    samples = collect_samples(args.annotation_root)
    output_dir = args.annotation_root / "closed_loop_verify"
    output_dir.mkdir(parents=True, exist_ok=True)
    draw_gt_visualizations(samples, output_dir / "gt_visualizations")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, criterion, feature_levels, grid_sizes = build_model_and_loss(config, args.checkpoint, device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)

    initial_loss = evaluate_loss(model, criterion, samples, config, feature_levels, grid_sizes, device)
    model.train()
    train_losses = [initial_loss]
    for sample in samples:
        image, targets = build_batch(sample, config, feature_levels, grid_sizes, device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(image), targets)["total_loss"]
        loss.backward()
        optimizer.step()
        train_losses.append(float(loss.item()))
    final_loss = evaluate_loss(model, criterion, samples, config, feature_levels, grid_sizes, device)
    train_losses.append(final_loss)
    curve_path = output_dir / "loss_curve.jpg"
    save_loss_curve(train_losses, curve_path)
    report = {
        "num_samples": len(samples),
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_decreased": final_loss <= initial_loss,
        "loss_curve": str(curve_path),
        "gt_visualization_dir": str(output_dir / "gt_visualizations"),
        "device": str(device),
    }
    report_path = output_dir / "closed_loop_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if final_loss > initial_loss:
        print("Warning: format is readable, but one epoch did not reduce loss on this tiny sample set.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
