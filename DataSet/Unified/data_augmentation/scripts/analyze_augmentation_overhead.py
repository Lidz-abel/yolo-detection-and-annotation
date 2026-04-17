from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze augmentation overhead results and generate plots.")
    parser.add_argument(
        "--input-json",
        type=Path,
        default=Path("/home/lidz/YOLO/DataSet/Unified/data_augmentation/results/augmentation_overhead.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/lidz/YOLO/DataSet/Unified/data_augmentation/plots"),
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_aggregated(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["aggregated"]


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def make_throughput_plot(rows: list[dict], batch_size: int, output_path: Path) -> None:
    subset = [row for row in rows if row["batch_size"] == batch_size]
    subset.sort(key=lambda row: row["samples_per_second_mean"], reverse=True)
    labels = [row["pipeline"] for row in subset]
    values = [row["samples_per_second_mean"] for row in subset]
    errors = [row["samples_per_second_std"] for row in subset]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(labels, values, yerr=errors, capsize=5)
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Samples / Second")
    plt.title(f"Augmentation Overhead Throughput Comparison (batch={batch_size})")
    plt.grid(axis="y", alpha=0.3)
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def make_ratio_plot(rows: list[dict], batch_size: int, output_path: Path) -> dict[str, float]:
    subset = [row for row in rows if row["batch_size"] == batch_size]
    baseline = next(row for row in subset if row["pipeline"] == "no_augmentation")
    baseline_sps = baseline["samples_per_second_mean"]
    subset.sort(key=lambda row: row["pipeline"])
    labels = [row["pipeline"] for row in subset]
    ratios = [row["samples_per_second_mean"] / baseline_sps if baseline_sps > 0 else 0.0 for row in subset]
    ratio_map = dict(zip(labels, ratios))

    plt.figure(figsize=(12, 6))
    bars = plt.bar(labels, ratios)
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Relative Throughput vs no_augmentation")
    plt.title(f"Augmentation Relative Throughput (batch={batch_size})")
    plt.grid(axis="y", alpha=0.3)
    for bar, value in zip(bars, ratios):
        plt.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return ratio_map


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    rows = load_aggregated(args.input_json)
    summary = {"by_batch": {}}

    for batch_size in sorted({row["batch_size"] for row in rows}):
        make_throughput_plot(rows, batch_size, args.output_dir / f"augmentation_throughput_batch_{batch_size}.png")
        ratios = make_ratio_plot(rows, batch_size, args.output_dir / f"augmentation_ratio_batch_{batch_size}.png")
        summary["by_batch"][str(batch_size)] = {
            "rows": [row for row in rows if row["batch_size"] == batch_size],
            "ratio_vs_no_augmentation": ratios,
        }

    save_json(args.output_dir / "augmentation_overhead_summary.json", summary)
    print(f"Wrote augmentation overhead plots to {args.output_dir}")


if __name__ == "__main__":
    main()
