"""Export checkpoint 6 version-02 figures from formal result records.

This script builds report-ready figures for the post-version01 work:
- Stage A/B full-loss experiments
- Stage C baseline and subsequent controlled rounds
- current Round 4A/4B visualization montages
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "report_assets" / "checkpoint06_v02"

RUN_RESULTS = {
    "stage_a": PROJECT_ROOT / "logs" / "records" / "deep_cnn_single_box_full_loss_20260419_020616" / "result.txt",
    "stage_b": PROJECT_ROOT / "logs" / "records" / "deep_residual_single_box_full_loss_20260419_031404" / "result.txt",
    "stage_c_base": PROJECT_ROOT / "logs" / "records" / "deep_residual_three_box_full_loss_20260419_043041" / "result.txt",
    "round1": PROJECT_ROOT / "logs" / "records" / "deep_residual_three_box_v5box_softobj_20260419_135531" / "result.txt",
    "round2": PROJECT_ROOT / "logs" / "records" / "deep_residual_three_box_v5box_softobj_assign_20260419_152713" / "result.txt",
    "round3b": PROJECT_ROOT / "logs" / "records" / "deep_residual_three_box_v5box_softobj_clamp_20260419_171735" / "result.txt",
    "round3c": PROJECT_ROOT / "logs" / "records" / "deep_residual_three_box_v5box_softobj_shapematch_20260419_184314" / "result.txt",
    "round4a": PROJECT_ROOT / "logs" / "records" / "deep_residual_three_box_v5box_softobj_shapematch_tight_20260419_201922" / "result.txt",
    "round4b": PROJECT_ROOT / "logs" / "records" / "deep_residual_three_box_v5box_softobj_shapematch_ignore_20260419_215025" / "result.txt",
}

EVAL_RESULTS = {
    "stage_a": PROJECT_ROOT / "outputs" / "evaluations" / "deep_cnn_single_box_full_loss_eval.json",
    "stage_b": PROJECT_ROOT / "outputs" / "evaluations" / "deep_residual_single_box_full_loss_eval.json",
    "stage_c_base": PROJECT_ROOT / "outputs" / "evaluations" / "deep_residual_three_box_full_loss_eval.json",
    "round1": PROJECT_ROOT / "outputs" / "evaluations" / "deep_residual_three_box_v5box_softobj_eval.json",
    "round2": PROJECT_ROOT / "outputs" / "evaluations" / "deep_residual_three_box_v5box_softobj_assign_eval.json",
    "round3b": PROJECT_ROOT / "outputs" / "evaluations" / "deep_residual_three_box_v5box_softobj_clamp_eval.json",
    "round3c": PROJECT_ROOT / "outputs" / "evaluations" / "deep_residual_three_box_v5box_softobj_shapematch_eval.json",
    "round4a": PROJECT_ROOT / "outputs" / "evaluations" / "deep_residual_three_box_v5box_softobj_shapematch_tight_eval.json",
    "round4b": PROJECT_ROOT / "outputs" / "evaluations" / "deep_residual_three_box_v5box_softobj_shapematch_ignore_eval.json",
}

COCO_RESULTS = {
    "stage_a": PROJECT_ROOT / "outputs" / "evaluations" / "deep_cnn_single_box_full_loss_coco_eval.json",
    "stage_b": PROJECT_ROOT / "outputs" / "evaluations" / "deep_residual_single_box_full_loss_coco_eval.json",
    "stage_c_base": PROJECT_ROOT / "outputs" / "evaluations" / "deep_residual_three_box_full_loss_coco_eval.json",
    "round1": PROJECT_ROOT / "outputs" / "evaluations" / "deep_residual_three_box_v5box_softobj_coco_eval.json",
    "round2": PROJECT_ROOT / "outputs" / "evaluations" / "deep_residual_three_box_v5box_softobj_assign_coco_eval.json",
    "round3b": PROJECT_ROOT / "outputs" / "evaluations" / "deep_residual_three_box_v5box_softobj_clamp_coco_eval.json",
    "round3c": PROJECT_ROOT / "outputs" / "evaluations" / "deep_residual_three_box_v5box_softobj_shapematch_coco_eval.json",
    "round4a": PROJECT_ROOT / "outputs" / "evaluations" / "deep_residual_three_box_v5box_softobj_shapematch_tight_coco_eval.json",
    "round4b": PROJECT_ROOT / "outputs" / "evaluations" / "deep_residual_three_box_v5box_softobj_shapematch_ignore_coco_eval.json",
}

DISPLAY_NAMES = {
    "stage_a": "Stage A",
    "stage_b": "Stage B",
    "stage_c_base": "Stage C Base",
    "round1": "Round 1",
    "round2": "Round 2",
    "round3b": "Round 3B",
    "round3c": "Round 3C",
    "round4a": "Round 4A",
    "round4b": "Round 4B",
}


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


def parse_result_file(path: Path) -> dict:
    """Parse one structured result file into scalars and epoch histories."""
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


def plot_stage_ab_loss_curves(results: dict[str, dict]) -> None:
    """Plot train/val total loss curves for Stage A and Stage B."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for axis, key in zip(axes, ["stage_a", "stage_b"]):
        train_hist = results[key]["train_history"]
        val_hist = results[key]["val_history"]
        epochs = [entry["epoch"] for entry in train_hist]
        axis.plot(epochs, [entry["total"] for entry in train_hist], marker="o", linewidth=2, label="Train total")
        axis.plot(epochs, [entry["total"] for entry in val_hist], marker="s", linewidth=2, label="Val total")
        axis.set_title(DISPLAY_NAMES[key])
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Loss")
        axis.grid(alpha=0.3)
        axis.legend()
    fig.suptitle("Stage A/B Total Loss Curves", y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "stage_ab_total_loss_curves.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_stage_ab_components(results: dict[str, dict]) -> None:
    """Plot validation box/obj/cls curves for Stage A and Stage B."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True)
    components = [("box", "Validation loss_box"), ("obj", "Validation loss_obj"), ("cls", "Validation loss_cls")]
    for axis, (component, title) in zip(axes, components):
        for key in ["stage_a", "stage_b"]:
            hist = results[key]["val_history"]
            epochs = [entry["epoch"] for entry in hist]
            axis.plot(epochs, [entry[component] for entry in hist], marker="o", linewidth=2, label=DISPLAY_NAMES[key])
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Loss")
        axis.grid(alpha=0.3)
        axis.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "stage_ab_component_curves.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_stage_c_loss_grid(results: dict[str, dict]) -> None:
    """Plot one train/val total-loss subplot for each formal Stage C experiment."""
    keys = ["stage_c_base", "round1", "round2", "round3b", "round3c", "round4a", "round4b"]
    fig, axes = plt.subplots(4, 2, figsize=(13, 16), sharex=False, sharey=False)
    axes = axes.flatten()
    for axis, key in zip(axes, keys):
        train_hist = results[key]["train_history"]
        val_hist = results[key]["val_history"]
        epochs = [entry["epoch"] for entry in train_hist]
        axis.plot(epochs, [entry["total"] for entry in train_hist], marker="o", linewidth=1.8, label="Train")
        axis.plot(epochs, [entry["total"] for entry in val_hist], marker="s", linewidth=1.8, label="Val")
        axis.set_title(DISPLAY_NAMES[key])
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Loss")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    axes[-1].axis("off")
    fig.suptitle("Stage C Formal Experiments: Total Loss Curves", y=0.995)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "stage_c_total_loss_grid.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_stage_c_internal_metrics() -> None:
    """Plot internal mAP/precision/recall and prediction count trends for Stage C."""
    keys = ["stage_c_base", "round1", "round2", "round3b", "round3c", "round4a", "round4b"]
    labels = [DISPLAY_NAMES[key] for key in keys]
    metrics = {key: json.loads(EVAL_RESULTS[key].read_text(encoding="utf-8")) for key in keys}

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(labels, [metrics[key]["map50"] for key in keys], marker="o", linewidth=2, label="mAP@0.5")
    axes[0].plot(labels, [metrics[key]["precision"] for key in keys], marker="s", linewidth=2, label="Precision")
    axes[0].plot(labels, [metrics[key]["recall"] for key in keys], marker="^", linewidth=2, label="Recall")
    axes[0].set_title("Internal Detection Metrics")
    axes[0].set_ylabel("Metric value")
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    axes[0].tick_params(axis="x", rotation=35)

    axes[1].plot(labels, [metrics[key]["num_predictions"] for key in keys], marker="o", linewidth=2, color="tab:red")
    axes[1].set_title("Prediction Count Trend")
    axes[1].set_ylabel("Predictions on full val set")
    axes[1].grid(alpha=0.3)
    axes[1].tick_params(axis="x", rotation=35)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "stage_c_internal_metrics.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_stage_c_coco_metrics() -> None:
    """Plot COCO AP50/AR100 trends for Stage C rounds."""
    keys = ["stage_c_base", "round1", "round2", "round3b", "round3c", "round4a", "round4b"]
    labels = [DISPLAY_NAMES[key] for key in keys]
    metrics = {key: json.loads(COCO_RESULTS[key].read_text(encoding="utf-8")) for key in keys}

    plt.figure(figsize=(10.5, 4.5))
    plt.plot(labels, [metrics[key]["coco_ap50"] for key in keys], marker="o", linewidth=2, label="COCO AP50")
    plt.plot(labels, [metrics[key]["coco_ar100"] for key in keys], marker="s", linewidth=2, label="COCO AR@100")
    plt.ylabel("Metric value")
    plt.title("Stage C COCO-subset Metrics")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.xticks(rotation=35)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "stage_c_coco_metrics.png", dpi=220, bbox_inches="tight")
    plt.close()


def make_visualization_montage(run_dir_name: str, output_name: str) -> None:
    """Build a 2x2 montage from the four fixed visualization samples."""
    image_paths = [
        PROJECT_ROOT / "outputs" / run_dir_name / "visualizations" / "voc2012_val_2008_000002.png",
        PROJECT_ROOT / "outputs" / run_dir_name / "visualizations" / "voc2012_val_2008_000003.png",
        PROJECT_ROOT / "outputs" / run_dir_name / "visualizations" / "voc2012_val_2008_000007.png",
        PROJECT_ROOT / "outputs" / run_dir_name / "visualizations" / "voc2012_val_2008_000009.png",
    ]
    labels = [path.stem for path in image_paths]
    images = [Image.open(path).convert("RGB") for path in image_paths]
    width, height = images[0].size
    pad = 20
    title_h = 30
    canvas = Image.new("RGB", (2 * width + 3 * pad, 2 * (height + title_h) + 3 * pad), "white")
    draw = ImageDraw.Draw(canvas)

    for index, (label, image) in enumerate(zip(labels, images)):
        row, col = divmod(index, 2)
        x = pad + col * (width + pad)
        y = pad + row * (height + title_h + pad)
        draw.text((x, y), label, fill="black")
        canvas.paste(ImageOps.expand(image, border=1, fill="black"), (x, y + title_h))

    canvas.save(OUTPUT_DIR / output_name)


def main() -> None:
    """Export all figures needed by checkpoint06 version02."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {name: parse_result_file(path) for name, path in RUN_RESULTS.items()}
    plot_stage_ab_loss_curves(results)
    plot_stage_ab_components(results)
    plot_stage_c_loss_grid(results)
    plot_stage_c_internal_metrics()
    plot_stage_c_coco_metrics()
    make_visualization_montage(
        "deep_residual_three_box_v5box_softobj_shapematch_tight_20260419_201922",
        "round4a_visualizations_2x2.png",
    )
    make_visualization_montage(
        "deep_residual_three_box_v5box_softobj_shapematch_ignore_20260419_215025",
        "round4b_visualizations_2x2.png",
    )

    summary = {
        "export_dir": str(OUTPUT_DIR),
        "run_results": {name: str(path) for name, path in RUN_RESULTS.items()},
        "eval_results": {name: str(path) for name, path in EVAL_RESULTS.items()},
        "coco_results": {name: str(path) for name, path in COCO_RESULTS.items()},
    }
    (OUTPUT_DIR / "figure_export_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
