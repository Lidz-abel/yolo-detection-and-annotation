#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from chunk_benchmark_lib import (
    FORMAT_EXTENSIONS,
    SUPPORTED_FORMATS,
    build_chunk_payload,
    build_index_payload,
    chunk_records,
    ensure_dir,
    load_manifest,
    save_chunk,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pack unified detection samples into chunked formats.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("/home/lidz/YOLO/DataSet/Unified/manifests/all_val.jsonl"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/home/lidz/YOLO/DataSet/Unified/packed"),
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=list(SUPPORTED_FORMATS),
        choices=SUPPORTED_FORMATS,
    )
    parser.add_argument("--chunk-sizes", nargs="+", type=int, default=[64, 256, 1024])
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_manifest(args.manifest, max_samples=args.max_samples)
    manifest_stem = args.manifest.stem

    print(f"Loaded {len(records)} samples from {args.manifest}")

    for fmt in args.formats:
        for chunk_size in args.chunk_sizes:
            chunk_dir = args.output_root / fmt / manifest_stem / f"chunk_{chunk_size}"
            ensure_dir(chunk_dir)

            chunk_files = []
            chunk_counts = []
            for chunk_idx, chunk in enumerate(chunk_records(records, chunk_size)):
                chunk_file = chunk_dir / f"chunk_{chunk_idx:05d}{FORMAT_EXTENSIONS[fmt]}"
                payload = build_chunk_payload(chunk)
                save_chunk(chunk_file, payload, fmt)
                chunk_files.append(chunk_file)
                chunk_counts.append(len(chunk))
                print(f"[{fmt}] chunk_size={chunk_size} wrote {chunk_file.name} with {len(chunk)} samples")

            index_payload = build_index_payload(
                fmt=fmt,
                manifest_path=args.manifest,
                chunk_size=chunk_size,
                max_samples=args.max_samples,
                chunk_files=chunk_files,
                chunk_sample_counts=chunk_counts,
            )
            index_path = chunk_dir / "index.json"
            index_path.write_text(json.dumps(index_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"[{fmt}] chunk_size={chunk_size} wrote index.json")


if __name__ == "__main__":
    main()
