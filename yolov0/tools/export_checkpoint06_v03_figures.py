"""Export Round 5 and Round 6 report figures for checkpoint06_version03."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = PROJECT_ROOT / "outputs" / "report_assets" / "checkpoint06_v03"


RUNS = {
    "5A": {
        "label": "Round 5A",
        "result": PROJECT_ROOT / "logs/records/deep_residual_three_box_qualitycls_20260420_001813/result.txt",
        "eval": PROJECT_ROOT / "outputs/evaluations/deep_residual_three_box_qualitycls_eval.json",
        "coco": PROJECT_ROOT / "outputs/evaluations/deep_residual_three_box_qualitycls_coco_eval.json",
        "vis_dir": PROJECT_ROOT / "outputs/deep_residual_three_box_qualitycls_20260420_001813/visualizations",
    },
    "5B": {
        "label": "Round 5B",
        "result": PROJECT_ROOT / "logs/records/deep_residual_three_box_qualitycls_decoupled_20260420_014124/result.txt",
        "eval": PROJECT_ROOT / "outputs/evaluations/deep_residual_three_box_qualitycls_decoupled_eval.json",
        "coco": PROJECT_ROOT / "outputs/evaluations/deep_residual_three_box_qualitycls_decoupled_coco_eval.json",
        "vis_dir": PROJECT_ROOT / "outputs/deep_residual_three_box_qualitycls_decoupled_20260420_014124/visualizations",
    },
    "5C": {
        "label": "Round 5C",
        "result": PROJECT_ROOT / "logs/records/deep_residual_three_box_dynamicassign_20260420_030501/result.txt",
        "eval": PROJECT_ROOT / "outputs/evaluations/deep_residual_three_box_dynamicassign_eval.json",
        "coco": PROJECT_ROOT / "outputs/evaluations/deep_residual_three_box_dynamicassign_coco_eval.json",
        "vis_dir": PROJECT_ROOT / "outputs/deep_residual_three_box_dynamicassign_20260420_030501/visualizations",
    },
    "6A": {
        "label": "Round 6A",
        "result": PROJECT_ROOT / "logs/records/deep_residual_three_box_dynamicassign_scoretune_20260421_182204/result.txt",
        "eval": PROJECT_ROOT / "outputs/evaluations/deep_residual_three_box_dynamicassign_scoretune_eval.json",
        "coco": PROJECT_ROOT / "outputs/evaluations/deep_residual_three_box_dynamicassign_scoretune_coco_eval.json",
        "vis_dir": PROJECT_ROOT / "outputs/deep_residual_three_box_dynamicassign_scoretune_20260421_182204/visualizations",
    },
    "6B": {
        "label": "Round 6B",
        "result": PROJECT_ROOT / "logs/records/deep_residual_three_box_dynamicassign_varifocal_20260424_110543/result.txt",
        "eval": PROJECT_ROOT / "outputs/evaluations/deep_residual_three_box_dynamicassign_varifocal_eval.json",
        "coco": PROJECT_ROOT / "outputs/evaluations/deep_residual_three_box_dynamicassign_varifocal_coco_eval.json",
        "vis_dir": PROJECT_ROOT / "outputs/deep_residual_three_box_dynamicassign_varifocal_20260424_110543/visualizations",
    },
    "6C": {
        "label": "Round 6C",
        "result": PROJECT_ROOT / "logs/records/deep_residual_three_box_dynamicassign_topk1_20260423_210757/result.txt",
        "eval": PROJECT_ROOT / "outputs/evaluations/deep_residual_three_box_dynamicassign_topk1_eval.json",
        "coco": PROJECT_ROOT / "outputs/evaluations/deep_residual_three_box_dynamicassign_topk1_coco_eval.json",
        "vis_dir": PROJECT_ROOT / "outputs/deep_residual_three_box_dynamicassign_topk1_20260423_210757/visualizations",
    },
}

VIS_FILES = [
    "voc2012_val_2008_000002.png",
    "voc2012_val_2008_000003.png",
    "voc2012_val_2008_000007.png",
    "voc2012_val_2008_000009.png",
]


def parse_history(result_path: Path, section_name: str) -> list[dict]:
    """Parse epoch metric lines from the text result file."""
    lines = result_path.read_text(encoding="utf-8").splitlines()
    capture = False
    rows: list[dict] = []
    for line in lines:
        stripped = line.strip()
        if stripped == f"{section_name}:":
            capture = True
            continue
        if capture and not stripped:
            break
        if capture and stripped.startswith("epoch ="):
            row = {}
            for part in stripped.split("|"):
                if "=" not in part:
                    continue
                key, value = [item.strip() for item in part.split("=", 1)]
                key = key.replace(" ", "_")
                try:
                    row[key] = float(value.rstrip("s"))
                except ValueError:
                    row[key] = value
            rows.append(row)
    return rows


def load_metric_bundle():
    """Load parsed histories and json metrics into one dictionary."""
    bundle = {}
    for key, cfg in RUNS.items():
        train_hist = parse_history(cfg["result"], "train history")
        val_hist = parse_history(cfg["result"], "val history")
        bundle[key] = {
            "label": cfg["label"],
            "train": train_hist,
            "val": val_hist,
            "eval": json.loads(cfg["eval"].read_text(encoding="utf-8")),
            "coco": json.loads(cfg["coco"].read_text(encoding="utf-8")),
            "vis_dir": cfg["vis_dir"],
        }
    return bundle


def plot_round56_loss_curves(bundle):
    """Plot train/val total loss curves for the training rounds."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    train_keys = ["5A", "5B", "5C", "6B", "6C"]

    for key in train_keys:
        item = bundle[key]
        if item["train"]:
            train_epochs = [row["epoch"] for row in item["train"]]
            train_total = [row["total"] for row in item["train"]]
            axes[0].plot(train_epochs, train_total, marker="o", linewidth=1.8, label=item["label"])
        if item["val"]:
            val_epochs = [row["epoch"] for row in item["val"]]
            val_total = [row["total"] for row in item["val"]]
            axes[1].plot(val_epochs, val_total, marker="o", linewidth=1.8, label=item["label"])

    axes[0].set_title("Round 5/6 Train Total Loss")
    axes[1].set_title("Round 5/6 Val Total Loss")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
    axes[1].legend(loc="best", fontsize=8)
    fig.savefig(ASSET_DIR / "round56_total_loss_curves.png", dpi=200)
    plt.close(fig)


def plot_round56_metric_bars(bundle):
    """Plot key internal and COCO metrics for Round 5/6."""
    keys = ["5A", "5B", "5C", "6A", "6B", "6C"]
    labels = [bundle[key]["label"] for key in keys]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    internal_map = [bundle[key]["eval"]["map50"] for key in keys]
    internal_precision = [bundle[key]["eval"]["precision"] for key in keys]
    internal_recall = [bundle[key]["eval"]["recall"] for key in keys]
    x = range(len(keys))
    width = 0.25
    axes[0].bar([i - width for i in x], internal_map, width=width, label="mAP@0.5")
    axes[0].bar(x, internal_precision, width=width, label="Precision")
    axes[0].bar([i + width for i in x], internal_recall, width=width, label="Recall")
    axes[0].set_xticks(list(x), labels, rotation=20)
    axes[0].set_title("Internal Metrics")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].legend(fontsize=8)

    coco_ap50 = [bundle[key]["coco"]["coco_ap50"] for key in keys]
    coco_ar100 = [bundle[key]["coco"]["coco_ar100"] for key in keys]
    axes[1].bar([i - width / 2 for i in x], coco_ap50, width=width, label="COCO AP50")
    axes[1].bar([i + width / 2 for i in x], coco_ar100, width=width, label="COCO AR@100")
    axes[1].set_xticks(list(x), labels, rotation=20)
    axes[1].set_title("COCO Subset Metrics")
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.savefig(ASSET_DIR / "round56_metric_bars.png", dpi=200)
    plt.close(fig)


def _resize_with_frame(path: Path, label: str, tile_size=(520, 390)) -> Image.Image:
    """Resize one visualization tile and draw a simple label."""
    image = Image.open(path).convert("RGB")
    image = ImageOps.contain(image, tile_size)
    canvas = Image.new("RGB", tile_size, color="white")
    offset = ((tile_size[0] - image.width) // 2, (tile_size[1] - image.height) // 2)
    canvas.paste(image, offset)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, tile_size[0] - 1, tile_size[1] - 1), outline="black", width=2)
    draw.rectangle((0, 0, tile_size[0], 28), fill="white")
    draw.text((10, 6), label, fill="black")
    return canvas


def export_visual_grid(bundle, key: str):
    """Export one 2x2 visualization grid for a given run."""
    vis_dir = bundle[key]["vis_dir"]
    tiles = []
    for filename in VIS_FILES:
        tiles.append(_resize_with_frame(vis_dir / filename, filename))
    tile_w, tile_h = tiles[0].size
    grid = Image.new("RGB", (tile_w * 2, tile_h * 2), color="white")
    positions = [(0, 0), (tile_w, 0), (0, tile_h), (tile_w, tile_h)]
    for tile, pos in zip(tiles, positions):
        grid.paste(tile, pos)
    grid.save(ASSET_DIR / f"{key.lower()}_visualizations_2x2.png")


def write_summary_json(bundle):
    """Write a compact export summary for traceability."""
    summary = {}
    for key, item in bundle.items():
        summary[key] = {
            "label": item["label"],
            "best_val_total": min(row["total"] for row in item["val"]) if item["val"] else None,
            "internal_map50": item["eval"]["map50"],
            "internal_precision": item["eval"]["precision"],
            "internal_recall": item["eval"]["recall"],
            "coco_ap50": item["coco"]["coco_ap50"],
            "coco_ar100": item["coco"]["coco_ar100"],
        }
    (ASSET_DIR / "figure_export_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main():
    """Generate all assets needed by checkpoint06_version03."""
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    bundle = load_metric_bundle()
    plot_round56_loss_curves(bundle)
    plot_round56_metric_bars(bundle)
    for key in bundle:
        export_visual_grid(bundle, key)
    write_summary_json(bundle)


if __name__ == "__main__":
    main()
