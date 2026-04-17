#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze stage1 loader scan results and generate plots.")
    parser.add_argument(
        "--input-json",
        type=Path,
        default=Path("/home/lidz/YOLO/DataSet/Unified/stage1_loader_scan.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/lidz/YOLO/DataSet/Unified/plots/stage1"),
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_aggregated(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["aggregated"]


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def rows_for(rows: list[dict], storage_mode: str, batch_size: int) -> list[dict]:
    return [
        row
        for row in rows
        if row["storage_mode"] == storage_mode and row["batch_size"] == batch_size
    ]


def best_row(rows: list[dict], storage_mode: str, batch_size: int) -> dict:
    subset = rows_for(rows, storage_mode, batch_size)
    return max(subset, key=lambda row: row["samples_per_second_mean"])


def make_num_workers_plot(rows: list[dict], output_path: Path, batch_size: int) -> None:
    plt.figure(figsize=(10, 6))
    for storage_mode, label in [("raw", "raw"), ("chunked", "hdf5 chunk=64")]:
        subset = [
            row
            for row in rows_for(rows, storage_mode, batch_size)
            if row["pin_memory"] is False
            and row["persistent_workers"] is False
            and row["prefetch_factor"] == 4
        ]
        subset.sort(key=lambda row: row["num_workers"])
        xs = [row["num_workers"] for row in subset]
        ys = [row["samples_per_second_mean"] for row in subset]
        plt.plot(xs, ys, marker="o", label=label)

    plt.xlabel("num_workers")
    plt.ylabel("Samples / Second")
    plt.title(f"Stage1: num_workers Impact on Throughput (batch={batch_size})")
    plt.xticks(sorted({row["num_workers"] for row in rows}))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def make_best_config_plot(rows: list[dict], output_path: Path, batch_size: int) -> None:
    best_raw = best_row(rows, "raw", batch_size)
    best_chunked = best_row(rows, "chunked", batch_size)
    labels = ["raw", "hdf5 chunk=64"]
    values = [best_raw["samples_per_second_mean"], best_chunked["samples_per_second_mean"]]
    errors = [best_raw["samples_per_second_std"], best_chunked["samples_per_second_std"]]

    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, values, yerr=errors, capsize=6)
    plt.ylabel("Samples / Second")
    plt.title(f"Stage1: Best Throughput by Loading Mode (batch={batch_size})")
    plt.grid(axis="y", alpha=0.3)
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)

    aggregated = load_aggregated(args.input_json)
    summary = {"best_by_batch": {}}

    for batch_size in sorted({row["batch_size"] for row in aggregated}):
        summary["best_by_batch"][str(batch_size)] = {
            "raw": best_row(aggregated, "raw", batch_size),
            "chunked": best_row(aggregated, "chunked", batch_size),
        }
        make_num_workers_plot(
            aggregated,
            args.output_dir / f"num_workers_throughput_batch_{batch_size}.png",
            batch_size=batch_size,
        )
        make_best_config_plot(
            aggregated,
            args.output_dir / f"best_loader_batch_{batch_size}.png",
            batch_size=batch_size,
        )

    save_json(args.output_dir / "summary.json", summary)
    print(f"Wrote stage1 plots and summary to {args.output_dir}")


if __name__ == "__main__":
    main()
