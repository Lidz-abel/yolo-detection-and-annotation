from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import time
from pathlib import Path

import psutil
import torch
from torch.utils.data import DataLoader

from detection_augmentation_dataset import AugmentedDetectionDataset
from detection_augmentations import (
    ColorJitterTransform,
    Compose,
    MixUp,
    Mosaic,
    RandomAffine,
    RandomCrop,
    RandomFlip,
    default_train_augmentation_pipeline,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark augmentation overhead in detection Dataset.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("/home/lidz/YOLO/DataSet/Unified/manifests/all_val.jsonl"),
    )
    parser.add_argument("--max-samples", type=int, default=1024)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[16, 256])
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--warmup-batches", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("/home/lidz/YOLO/DataSet/Unified/data_augmentation/results/augmentation_overhead.csv"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("/home/lidz/YOLO/DataSet/Unified/data_augmentation/results/augmentation_overhead.json"),
    )
    return parser.parse_args()


def detection_collate(batch: list[dict]) -> dict:
    return {
        "images": [sample["image"] for sample in batch],
        "targets": [sample["target"] for sample in batch],
        "sample_ids": [sample["sample_id"] for sample in batch],
    }


def move_batch_to_device(batch: dict, device: str) -> dict:
    batch["images"] = [image.to(device, non_blocking=True) for image in batch["images"]]
    return batch


def get_gpu_snapshot(device: str) -> dict:
    if not device.startswith("cuda"):
        return {}
    try:
        index = int(device.split(":")[1]) if ":" in device else 0
        cmd = [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
        output = subprocess.check_output(cmd, text=True)
        for line in output.strip().splitlines():
            gpu_index, gpu_util, mem_used, mem_total = [part.strip() for part in line.split(",")]
            if int(gpu_index) == index:
                return {
                    "gpu_index": int(gpu_index),
                    "gpu_util_percent": float(gpu_util),
                    "gpu_memory_used_mib": float(mem_used),
                    "gpu_memory_total_mib": float(mem_total),
                }
    except Exception:
        return {}
    return {}


def benchmark_loader(
    dataset: AugmentedDetectionDataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int | None,
    max_batches: int,
    warmup_batches: int,
    device: str,
) -> dict:
    cuda_index = 0
    if device.startswith("cuda") and ":" in device:
        cuda_index = int(device.split(":")[1])

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=detection_collate,
        persistent_workers=persistent_workers and num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )

    if device.startswith("cuda"):
        torch.cuda.set_device(cuda_index)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    process = psutil.Process(os.getpid())
    rss_before = process.memory_info().rss
    cpu_percent_before = psutil.cpu_percent(interval=None)

    iterator = iter(loader)
    for _ in range(warmup_batches):
        try:
            batch = next(iterator)
        except StopIteration:
            break
        if device.startswith("cuda"):
            batch = move_batch_to_device(batch, device)
            torch.cuda.synchronize()

    total_samples = 0
    batch_count = 0
    first_batch_seconds = None
    start = time.perf_counter()

    while True:
        batch_start = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            break
        batch_end = time.perf_counter()

        if first_batch_seconds is None:
            first_batch_seconds = batch_end - batch_start

        if device.startswith("cuda"):
            batch = move_batch_to_device(batch, device)
            torch.cuda.synchronize()

        total_samples += len(batch["images"])
        batch_count += 1
        if max_batches is not None and batch_count >= max_batches:
            break

    total_seconds = time.perf_counter() - start
    cpu_percent_after = psutil.cpu_percent(interval=None)
    rss_after = process.memory_info().rss

    metrics = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers and num_workers > 0,
        "prefetch_factor": prefetch_factor if num_workers > 0 else None,
        "warmup_batches": warmup_batches,
        "max_batches": max_batches,
        "observed_batches": batch_count,
        "observed_samples": total_samples,
        "first_batch_seconds": first_batch_seconds if first_batch_seconds is not None else 0.0,
        "total_seconds": total_seconds,
        "samples_per_second": (total_samples / total_seconds) if total_seconds > 0 else 0.0,
        "process_rss_before_mb": rss_before / (1024 * 1024),
        "process_rss_after_mb": rss_after / (1024 * 1024),
        "process_rss_delta_mb": (rss_after - rss_before) / (1024 * 1024),
        "cpu_percent_before": cpu_percent_before,
        "cpu_percent_after": cpu_percent_after,
    }
    if device.startswith("cuda"):
        metrics["torch_cuda_max_memory_mb"] = torch.cuda.max_memory_allocated() / (1024 * 1024)
        metrics.update(get_gpu_snapshot(device))
    return metrics


def aggregate_results(results: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = {}
    for row in results:
        key = (
            row["pipeline"],
            row["batch_size"],
            row["num_workers"],
            row["pin_memory"],
            row["persistent_workers"],
            row["prefetch_factor"],
        )
        grouped.setdefault(key, []).append(row)

    aggregated = []
    metric_names = [
        "first_batch_seconds",
        "total_seconds",
        "samples_per_second",
        "process_rss_before_mb",
        "process_rss_after_mb",
        "process_rss_delta_mb",
        "cpu_percent_before",
        "cpu_percent_after",
        "torch_cuda_max_memory_mb",
        "gpu_util_percent",
        "gpu_memory_used_mib",
        "gpu_memory_total_mib",
    ]
    for rows in grouped.values():
        base = {
            key: rows[0][key]
            for key in [
                "pipeline",
                "batch_size",
                "num_workers",
                "pin_memory",
                "persistent_workers",
                "prefetch_factor",
                "warmup_batches",
                "max_batches",
            ]
        }
        base["repeats"] = len(rows)
        base["observed_batches"] = rows[0]["observed_batches"]
        base["observed_samples"] = rows[0]["observed_samples"]
        for metric_name in metric_names:
            values = [row[metric_name] for row in rows if metric_name in row and row[metric_name] is not None]
            if not values:
                continue
            base[f"{metric_name}_mean"] = statistics.mean(values)
            base[f"{metric_name}_std"] = statistics.pstdev(values) if len(values) > 1 else 0.0
        aggregated.append(base)
    return aggregated


def build_pipelines() -> dict[str, Compose | None]:
    return {
        "no_augmentation": None,
        "random_flip": Compose([RandomFlip(p_horizontal=1.0)]),
        "color_jitter": Compose([ColorJitterTransform(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.08, p=1.0)]),
        "random_affine": Compose([RandomAffine(degrees=12.0, translate=(0.08, 0.08), scale=(0.9, 1.1), shear=(-4.0, 4.0), p=1.0)]),
        "random_crop": Compose([RandomCrop(min_scale=0.65, p=1.0)]),
        "mixup": Compose([MixUp(alpha=0.4, p=1.0)]),
        "mosaic": Compose([Mosaic(output_size=(640, 640), p=1.0)]),
        "default_train_pipeline": default_train_augmentation_pipeline(),
    }


def main() -> None:
    args = parse_args()
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)

    results = []
    pipelines = build_pipelines()
    for pipeline_name, augment in pipelines.items():
        dataset = AugmentedDetectionDataset(
            manifest_path=args.manifest,
            max_samples=args.max_samples,
            image_mode="rgb",
            augment=augment,
            enable_mixup_mosaic=True,
        )
        for batch_size in args.batch_sizes:
            for repeat_idx in range(args.repeats):
                metrics = benchmark_loader(
                    dataset=dataset,
                    batch_size=batch_size,
                    num_workers=args.num_workers,
                    pin_memory=args.pin_memory,
                    persistent_workers=args.persistent_workers,
                    prefetch_factor=args.prefetch_factor,
                    max_batches=args.max_batches,
                    warmup_batches=args.warmup_batches,
                    device=args.device,
                )
                metrics.update(
                    {
                        "pipeline": pipeline_name,
                        "repeat_index": repeat_idx,
                    }
                )
                print(metrics)
                results.append(metrics)

    aggregated = aggregate_results(results)
    fieldnames = [
        "pipeline",
        "batch_size",
        "num_workers",
        "pin_memory",
        "persistent_workers",
        "prefetch_factor",
        "warmup_batches",
        "max_batches",
        "repeat_index",
        "observed_batches",
        "observed_samples",
        "first_batch_seconds",
        "total_seconds",
        "samples_per_second",
        "process_rss_before_mb",
        "process_rss_after_mb",
        "process_rss_delta_mb",
        "cpu_percent_before",
        "cpu_percent_after",
        "torch_cuda_max_memory_mb",
        "gpu_index",
        "gpu_util_percent",
        "gpu_memory_used_mib",
        "gpu_memory_total_mib",
    ]
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    args.output_json.write_text(
        json.dumps({"runs": results, "aggregated": aggregated}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
