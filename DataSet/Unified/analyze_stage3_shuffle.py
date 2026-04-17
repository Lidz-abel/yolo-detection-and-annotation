#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze stage3 shuffle benchmark results and generate plots.")
    parser.add_argument(
        "--input-json",
        type=Path,
        default=Path("/home/lidz/YOLO/DataSet/Unified/stage3_shuffle_benchmark.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/lidz/YOLO/DataSet/Unified/plots/stage3"),
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_aggregated(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["aggregated"]


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def label_for(row: dict) -> str:
    if row["format"] == "raw":
        return "raw"
    return f"{row['format']}-{row['chunk_size']}"


def make_shuffle_bar_plot(rows: list[dict], output_path: Path, batch_size: int) -> None:
    subset = [row for row in rows if row["batch_size"] == batch_size]
    groups = {}
    for row in subset:
        groups.setdefault(label_for(row), {})[row["shuffle"]] = row

    labels = list(groups.keys())
    false_values = [groups[label][False]["samples_per_second_mean"] for label in labels]
    true_values = [groups[label][True]["samples_per_second_mean"] for label in labels]

    x = range(len(labels))
    width = 0.38
    plt.figure(figsize=(11, 6))
    plt.bar([i - width / 2 for i in x], false_values, width=width, label="shuffle=False")
    plt.bar([i + width / 2 for i in x], true_values, width=width, label="shuffle=True")
    plt.xticks(list(x), labels, rotation=15)
    plt.ylabel("Samples / Second")
    plt.title(f"Stage3: Shuffle Impact on Throughput (batch={batch_size})")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def make_shuffle_ratio_plot(rows: list[dict], output_path: Path, batch_size: int) -> dict:
    subset = [row for row in rows if row["batch_size"] == batch_size]
    groups = {}
    ratios = {}
    for row in subset:
        groups.setdefault(label_for(row), {})[row["shuffle"]] = row

    labels = list(groups.keys())
    values = []
    for label in labels:
        no_shuffle = groups[label][False]["samples_per_second_mean"]
        shuffle = groups[label][True]["samples_per_second_mean"]
        ratio = shuffle / no_shuffle if no_shuffle > 0 else 0.0
        values.append(ratio)
        ratios[label] = ratio

    plt.figure(figsize=(11, 6))
    bars = plt.bar(labels, values)
    plt.xticks(rotation=15)
    plt.ylabel("shuffle=True / shuffle=False")
    plt.title(f"Stage3: Shuffle Throughput Ratio (batch={batch_size})")
    plt.grid(axis="y", alpha=0.3)
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return ratios


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)

    aggregated = load_aggregated(args.input_json)
    summary = {"by_batch": {}}

    for batch_size in sorted({row["batch_size"] for row in aggregated}):
        make_shuffle_bar_plot(
            aggregated,
            args.output_dir / f"shuffle_throughput_batch_{batch_size}.png",
            batch_size=batch_size,
        )
        ratios = make_shuffle_ratio_plot(
            aggregated,
            args.output_dir / f"shuffle_ratio_batch_{batch_size}.png",
            batch_size=batch_size,
        )
        summary["by_batch"][str(batch_size)] = {
            "rows": [row for row in aggregated if row["batch_size"] == batch_size],
            "shuffle_ratio": ratios,
        }

    save_json(args.output_dir / "summary.json", summary)
    print(f"Wrote stage3 plots and summary to {args.output_dir}")


if __name__ == "__main__":
    main()
