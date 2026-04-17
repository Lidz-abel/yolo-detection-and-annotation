#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze benchmark JSON results and generate plots.")
    parser.add_argument(
        "--input-json",
        type=Path,
        default=Path("/home/lidz/YOLO/DataSet/Unified/stage2_format_chunk_benchmark.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/lidz/YOLO/DataSet/Unified/plots"),
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_aggregated(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["aggregated"]


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def filter_rows(rows: list[dict], batch_size: int) -> list[dict]:
    return [row for row in rows if row["batch_size"] == batch_size]


def best_by_batch(rows: list[dict]) -> dict:
    result = {}
    for batch_size in sorted({row["batch_size"] for row in rows}):
        subset = filter_rows(rows, batch_size)
        result[str(batch_size)] = max(subset, key=lambda row: row["samples_per_second_mean"])
    return result


def make_chunk_plot(rows: list[dict], output_path: Path, batch_size: int) -> None:
    subset = [row for row in filter_rows(rows, batch_size) if row["storage_mode"] == "chunked"]
    formats = sorted({row["format"] for row in subset})
    chunk_sizes = sorted({row["chunk_size"] for row in subset})

    plt.figure(figsize=(10, 6))
    for fmt in formats:
        fmt_rows = sorted(
            [row for row in subset if row["format"] == fmt],
            key=lambda row: row["chunk_size"],
        )
        xs = [row["chunk_size"] for row in fmt_rows]
        ys = [row["samples_per_second_mean"] for row in fmt_rows]
        plt.plot(xs, ys, marker="o", label=fmt)

    raw_row = max(
        [row for row in filter_rows(rows, batch_size) if row["format"] == "raw"],
        key=lambda row: row["samples_per_second_mean"],
    )
    plt.axhline(
        raw_row["samples_per_second_mean"],
        color="black",
        linestyle="--",
        label=f"raw ({raw_row['samples_per_second_mean']:.1f})",
    )
    plt.xticks(chunk_sizes, chunk_sizes)
    plt.xlabel("Chunk Size")
    plt.ylabel("Samples / Second")
    plt.title(f"Chunk Size vs Loading Throughput (batch={batch_size})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def make_format_bar_plot(rows: list[dict], output_path: Path, batch_size: int) -> None:
    subset = filter_rows(rows, batch_size)
    best_rows = {}
    for row in subset:
        fmt = row["format"]
        if fmt not in best_rows or row["samples_per_second_mean"] > best_rows[fmt]["samples_per_second_mean"]:
            best_rows[fmt] = row

    labels = list(best_rows.keys())
    values = [best_rows[label]["samples_per_second_mean"] for label in labels]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, values)
    plt.ylabel("Best Samples / Second")
    plt.title(f"Best Throughput by Format (batch={batch_size})")
    plt.grid(axis="y", alpha=0.3)
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def make_memory_bar_plot(rows: list[dict], output_path: Path, batch_size: int) -> None:
    subset = filter_rows(rows, batch_size)
    best_rows = {}
    for row in subset:
        fmt = row["format"]
        if fmt not in best_rows or row["samples_per_second_mean"] > best_rows[fmt]["samples_per_second_mean"]:
            best_rows[fmt] = row

    labels = list(best_rows.keys())
    values = [best_rows[label].get("process_rss_after_mb_mean", 0.0) for label in labels]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, values)
    plt.ylabel("Process RSS After Benchmark (MB)")
    plt.title(f"Memory Footprint of Best Format Configurations (batch={batch_size})")
    plt.grid(axis="y", alpha=0.3)
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.0f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)

    aggregated = load_aggregated(args.input_json)
    summary = {"best_by_batch": best_by_batch(aggregated)}

    for batch_size in sorted({row["batch_size"] for row in aggregated}):
        make_chunk_plot(
            aggregated,
            args.output_dir / f"chunk_throughput_batch_{batch_size}.png",
            batch_size=batch_size,
        )
        make_format_bar_plot(
            aggregated,
            args.output_dir / f"best_format_batch_{batch_size}.png",
            batch_size=batch_size,
        )
        make_memory_bar_plot(
            aggregated,
            args.output_dir / f"memory_best_format_batch_{batch_size}.png",
            batch_size=batch_size,
        )

    save_json(args.output_dir / "summary.json", summary)
    print(f"Wrote plots and summary to {args.output_dir}")


if __name__ == "__main__":
    main()
