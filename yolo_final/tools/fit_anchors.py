"""Fit anchor boxes from the unified training manifest for stage-C experiments."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch


def parse_args():
    """Parse the manifest path, anchor count, and reference image size."""
    parser = argparse.ArgumentParser(description="Fit normalized anchors from detection boxes.")
    parser.add_argument("--manifest", type=str, required=True, help="Path to the training manifest jsonl.")
    parser.add_argument("--num-anchors", type=int, default=3, help="Number of anchors to fit.")
    parser.add_argument("--image-size", type=int, default=320, help="Reference image size for pixel reporting.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for centroid initialization.")
    return parser.parse_args()


def load_wh_pairs(manifest_path: Path) -> torch.Tensor:
    """Load normalized box width/height pairs directly from the unified manifest."""
    wh_pairs = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            image_width = float(sample["width"])
            image_height = float(sample["height"])
            for x1, y1, x2, y2 in sample["boxes"]:
                box_w = max(float(x2) - float(x1), 1.0) / image_width
                box_h = max(float(y2) - float(y1), 1.0) / image_height
                wh_pairs.append([box_w, box_h])
    if not wh_pairs:
        raise ValueError("No boxes were found in the manifest.")
    return torch.tensor(wh_pairs, dtype=torch.float32)


def wh_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Compute IoU on width/height pairs assuming the same box center."""
    inter_w = torch.minimum(boxes1[:, None, 0], boxes2[None, :, 0])
    inter_h = torch.minimum(boxes1[:, None, 1], boxes2[None, :, 1])
    intersection = inter_w * inter_h
    area1 = boxes1[:, None, 0] * boxes1[:, None, 1]
    area2 = boxes2[None, :, 0] * boxes2[None, :, 1]
    union = area1 + area2 - intersection
    return intersection / union.clamp(min=1e-7)


def init_centroids(points: torch.Tensor, num_anchors: int, seed: int) -> torch.Tensor:
    """Initialize k-means centers with a simple distance-aware sampling scheme."""
    random.seed(seed)
    first_index = random.randrange(points.shape[0])
    centroids = [points[first_index]]

    while len(centroids) < num_anchors:
        current = torch.stack(centroids, dim=0)
        ious = wh_iou(points, current)
        distances = 1.0 - ious.max(dim=1).values
        next_index = int(torch.argmax(distances).item())
        centroids.append(points[next_index])

    return torch.stack(centroids, dim=0)


def fit_anchors(points: torch.Tensor, num_anchors: int, seed: int, max_iters: int = 100) -> torch.Tensor:
    """Fit anchors with k-means under the width/height IoU distance."""
    centroids = init_centroids(points, num_anchors, seed)

    for _ in range(max_iters):
        ious = wh_iou(points, centroids)
        assignments = torch.argmax(ious, dim=1)

        new_centroids = []
        for anchor_index in range(num_anchors):
            members = points[assignments == anchor_index]
            if members.numel() == 0:
                new_centroids.append(centroids[anchor_index])
            else:
                new_centroids.append(members.mean(dim=0))
        new_centroids = torch.stack(new_centroids, dim=0)

        if torch.allclose(new_centroids, centroids, atol=1e-6):
            centroids = new_centroids
            break
        centroids = new_centroids

    anchor_areas = centroids[:, 0] * centroids[:, 1]
    order = torch.argsort(anchor_areas)
    return centroids[order]


def format_anchor_string(anchors: torch.Tensor) -> str:
    """Convert anchors into a compact TOML-friendly string representation."""
    return ";".join(f"{w:.6f},{h:.6f}" for w, h in anchors.tolist())


def main():
    """Fit anchors and print both normalized and pixel-space summaries."""
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    points = load_wh_pairs(manifest_path)
    anchors = fit_anchors(points, args.num_anchors, args.seed)

    print("manifest:", manifest_path)
    print("num_boxes:", args.num_anchors)
    print("normalized anchors:")
    for index, (w, h) in enumerate(anchors.tolist(), start=1):
        print(f"  {index}: w={w:.6f}, h={h:.6f}")

    print(f"pixel anchors @ image_size={args.image_size}:")
    for index, (w, h) in enumerate(anchors.tolist(), start=1):
        print(f"  {index}: w={w * args.image_size:.2f}, h={h * args.image_size:.2f}")

    print("anchor_string:", format_anchor_string(anchors))


if __name__ == "__main__":
    main()
