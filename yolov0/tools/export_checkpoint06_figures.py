"""Export the main checkpoint 6 figures used by the first summary report."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "report_assets" / "checkpoint06_v01"


RUNS = {
    "baseline_cnn": PROJECT_ROOT / "logs" / "records" / "baseline_formal_20260418_202723" / "result.txt",
    "deep_cnn": PROJECT_ROOT / "logs" / "records" / "deep_cnn_formal_20260418_213605" / "result.txt",
    "residual_small": PROJECT_ROOT / "logs" / "records" / "residual_small_formal_20260418_230104" / "result.txt",
    "resnet18_like": PROJECT_ROOT / "logs" / "records" / "resnet18_like_formal_20260419_002723" / "result.txt",
    "stage_a": PROJECT_ROOT / "logs" / "records" / "deep_cnn_single_box_full_loss_20260419_020616" / "result.txt",
    "stage_b": PROJECT_ROOT / "logs" / "records" / "deep_residual_single_box_full_loss_20260419_031404" / "result.txt",
    "stage_c": PROJECT_ROOT / "logs" / "records" / "deep_residual_three_box_full_loss_20260419_043041" / "result.txt",
}

COCO_EVALS = {
    "stage_a": PROJECT_ROOT / "outputs" / "evaluations" / "deep_cnn_single_box_full_loss_coco_eval.json",
    "stage_b": PROJECT_ROOT / "outputs" / "evaluations" / "deep_residual_single_box_full_loss_coco_eval.json",
    "stage_c": PROJECT_ROOT / "outputs" / "evaluations" / "deep_residual_three_box_full_loss_coco_eval.json",
}


def parse_result_file(path: Path) -> dict:
    """Parse one structured result.txt into scalars plus train/val histories."""
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    data: dict[str, object] = {"train_history": [], "val_history": []}
    section = None
    history_pattern = re.compile(r"([A-Za-z_]+) = ([^|]+)")

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        if line == "train history:":
            section = "train_history"
            continue
        if line == "val history:":
            section = "val_history"
            continue
        if line == "artifacts:":
            section = "artifacts"
            continue
        if line.endswith("summary:"):
            section = None
            continue

        if section in {"train_history", "val_history"} and line.startswith("epoch ="):
            entry = {}
            for key, value in history_pattern.findall(line):
                entry[key.strip()] = _convert_value(value.strip())
            data[section].append(entry)
            continue

        if section == "artifacts" and "=" in line:
            key, value = [part.strip() for part in line.split("=", 1)]
            data[key] = _convert_value(value)
            continue

        if "=" in line and section is None:
            key, value = [part.strip() for part in line.split("=", 1)]
            data[key] = _convert_value(value)

    return data


def _convert_value(value: str):
    """Convert plain strings into numbers when possible."""
    value = value.strip()
    if value.startswith("(") and value.endswith(")"):
        return value
    if value in {"True", "False"}:
        return value == "True"
    try:
        if any(ch in value for ch in [".", "e", "E"]):
            return float(value)
        return int(value)
    except ValueError:
        return value


def plot_backbone_val_curves(results: dict[str, dict]) -> None:
    """Plot the validation-total curves for the four backbone experiments."""
    plt.figure(figsize=(8, 5))
    for label in ["baseline_cnn", "deep_cnn", "residual_small", "resnet18_like"]:
        history = results[label]["val_history"]
        epochs = [entry["epoch"] for entry in history]
        totals = [entry["total"] for entry in history]
        plt.plot(epochs, totals, marker="o", linewidth=2, label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.title("Backbone Comparison Under Simplified MSE Loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "backbone_val_loss_curves.png", dpi=220)
    plt.close()


def plot_stage_total_curves(results: dict[str, dict]) -> None:
    """Plot train and val total losses for Stage A/B/C."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    mapping = {"stage_a": "Stage A", "stage_b": "Stage B", "stage_c": "Stage C"}
    for key, display in mapping.items():
        train_hist = results[key]["train_history"]
        val_hist = results[key]["val_history"]
        epochs = [entry["epoch"] for entry in train_hist]
        axes[0].plot(epochs, [entry["total"] for entry in train_hist], marker="o", linewidth=2, label=display)
        axes[1].plot(epochs, [entry["total"] for entry in val_hist], marker="o", linewidth=2, label=display)
    axes[0].set_title("Train Total Loss")
    axes[1].set_title("Validation Total Loss")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Loss")
        axis.grid(alpha=0.3)
        axis.legend()
    fig.suptitle("Stage A/B/C Total Loss Curves", y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "stage_total_loss_curves.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_stage_component_curves(results: dict[str, dict]) -> None:
    """Plot box, objectness, and classification validation losses for A/B/C."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True)
    mapping = {"stage_a": "Stage A", "stage_b": "Stage B", "stage_c": "Stage C"}
    components = [("box", "Validation loss_box"), ("obj", "Validation loss_obj"), ("cls", "Validation loss_cls")]
    for axis, (component, title) in zip(axes, components):
        for key, display in mapping.items():
            val_hist = results[key]["val_history"]
            epochs = [entry["epoch"] for entry in val_hist]
            axis.plot(epochs, [entry[component] for entry in val_hist], marker="o", linewidth=2, label=display)
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Loss")
        axis.grid(alpha=0.3)
        axis.legend()
    fig.suptitle("Stage A/B/C Validation Loss Components", y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "stage_component_curves.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_stage_giou_curves(results: dict[str, dict]) -> None:
    """Plot validation mean GIoU evolution for the full-loss runs."""
    plt.figure(figsize=(8, 5))
    mapping = {"stage_a": "Stage A", "stage_b": "Stage B", "stage_c": "Stage C"}
    for key, display in mapping.items():
        val_hist = results[key]["val_history"]
        epochs = [entry["epoch"] for entry in val_hist]
        plt.plot(epochs, [entry["giou"] for entry in val_hist], marker="o", linewidth=2, label=display)
    plt.xlabel("Epoch")
    plt.ylabel("Mean GIoU")
    plt.title("Validation Mean GIoU for Full-Loss Stages")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "stage_giou_curves.png", dpi=220)
    plt.close()


def plot_coco_bars() -> None:
    """Plot COCO-subset AP50 and AR100 plus efficiency bars for A/B/C."""
    names = []
    ap50 = []
    ar100 = []
    fps = []
    gflops = []
    params_m = []
    for key in ["stage_a", "stage_b", "stage_c"]:
        names.append({"stage_a": "Stage A", "stage_b": "Stage B", "stage_c": "Stage C"}[key])
        metrics = json.loads(COCO_EVALS[key].read_text(encoding="utf-8"))
        ap50.append(metrics["coco_ap50"])
        ar100.append(metrics["coco_ar100"])
        fps.append(metrics["images_per_second"])
        gflops.append(metrics["total_gflops"])
        params_m.append(metrics["params_total"] / 1e6)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    x = range(len(names))
    width = 0.35
    axes[0].bar([v - width / 2 for v in x], ap50, width=width, label="AP50")
    axes[0].bar([v + width / 2 for v in x], ar100, width=width, label="AR@100")
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(names)
    axes[0].set_title("COCO-subset Detection Metrics")
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].legend()

    axes[1].bar([v - width / 2 for v in x], fps, width=width, label="FPS")
    axes[1].bar([v + width / 2 for v in x], gflops, width=width, label="GFLOPs")
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(names)
    axes[1].set_title("COCO-subset Efficiency Metrics")
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "coco_metrics_bars.png", dpi=220)
    plt.close(fig)

    plt.figure(figsize=(7.5, 4.5))
    plt.bar(names, params_m)
    plt.ylabel("Parameters (Millions)")
    plt.title("Model Size for Stage A/B/C")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "stage_params_bar.png", dpi=220)
    plt.close()


def make_visualization_triptych(sample_name: str = "voc2012_val_2008_000002.png") -> None:
    """Combine the same validation sample from Stage A/B/C into one figure."""
    stage_paths = [
        PROJECT_ROOT / "outputs" / "deep_cnn_single_box_full_loss_20260419_020616" / "visualizations" / sample_name,
        PROJECT_ROOT / "outputs" / "deep_residual_single_box_full_loss_20260419_031404" / "visualizations" / sample_name,
        PROJECT_ROOT / "outputs" / "deep_residual_three_box_full_loss_20260419_043041" / "visualizations" / sample_name,
    ]
    labels = ["Stage A", "Stage B", "Stage C"]
    images = [Image.open(path).convert("RGB") for path in stage_paths]
    width, height = images[0].size
    pad = 20
    title_h = 40
    canvas = Image.new("RGB", (3 * width + 4 * pad, height + title_h + 2 * pad), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, (label, image) in enumerate(zip(labels, images)):
        x = pad + idx * (width + pad)
        y = pad + title_h
        canvas.paste(ImageOps.expand(image, border=1, fill="black"), (x, y))
        draw.text((x, pad), label, fill="black")
    canvas.save(OUTPUT_DIR / "stage_visualization_triptych.png")


def main() -> None:
    """Parse finished runs and export the main report figures."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {name: parse_result_file(path) for name, path in RUNS.items()}
    plot_backbone_val_curves(results)
    plot_stage_total_curves(results)
    plot_stage_component_curves(results)
    plot_stage_giou_curves(results)
    plot_coco_bars()
    make_visualization_triptych()
    summary = {
        "export_dir": str(OUTPUT_DIR),
        "runs": {name: str(path) for name, path in RUNS.items()},
        "coco_evals": {name: str(path) for name, path in COCO_EVALS.items()},
    }
    (OUTPUT_DIR / "figure_export_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
