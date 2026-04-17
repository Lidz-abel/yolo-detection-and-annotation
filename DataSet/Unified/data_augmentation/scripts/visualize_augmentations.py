from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from torchvision.io import ImageReadMode, read_image

from detection_augmentations import (
    ColorJitterTransform,
    MixUp,
    Mosaic,
    RandomAffine,
    RandomCrop,
    RandomFlip,
    sanitize_boxes,
)


def load_manifest(manifest_path: str | Path, max_samples: int | None = None) -> list[dict[str, Any]]:
    manifest_path = Path(manifest_path)
    records: list[dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            records.append(json.loads(line))
            if max_samples is not None and len(records) >= max_samples:
                break
    return records


def build_sample(record: dict[str, Any]) -> dict[str, Any]:
    image = read_image(record["image_path"], mode=ImageReadMode.RGB)
    boxes = torch.tensor(record["boxes"], dtype=torch.float32).reshape(-1, 4)
    labels = torch.tensor(record["labels"], dtype=torch.long)
    boxes, labels = sanitize_boxes(boxes, labels, int(record["height"]), int(record["width"]))
    return {
        "sample_id": record["sample_id"],
        "image": image,
        "boxes": boxes,
        "labels": labels,
        "meta": {
            "image_path": record["image_path"],
            "image_id": record["image_id"],
            "dataset_source": record["dataset_source"],
        },
    }


def draw_sample(ax: Any, sample: dict[str, Any], title: str) -> None:
    image = sample["image"].permute(1, 2, 0).cpu().numpy()
    ax.imshow(image)
    ax.set_title(title)
    ax.axis("off")
    for box, label in zip(sample["boxes"], sample["labels"]):
        x1, y1, x2, y2 = [float(v) for v in box.tolist()]
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor="lime", facecolor="none")
        ax.add_patch(rect)
        ax.text(x1, max(0.0, y1 - 2.0), str(int(label.item())), color="yellow", fontsize=8, backgroundcolor="black")


def save_comparison(before: dict[str, Any], after: dict[str, Any], output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    draw_sample(axes[0], before, "Before")
    draw_sample(axes[1], after, "After")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    manifest_path = Path("/home/lidz/YOLO/DataSet/Unified/manifests/all_val.jsonl")
    output_root = Path("/home/lidz/YOLO/DataSet/Unified/data_augmentation/visualizations")
    records = load_manifest(manifest_path, max_samples=32)
    random.seed(20260417)
    torch.manual_seed(20260417)

    base_samples = [build_sample(record) for record in records[:8]]

    def provider() -> dict[str, Any]:
        return build_sample(random.choice(records))

    transforms: list[tuple[str, Any]] = [
        ("random_flip", RandomFlip(p_horizontal=1.0, p_vertical=0.0)),
        ("color_jitter", ColorJitterTransform(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.08, p=1.0)),
        ("random_affine", RandomAffine(degrees=12.0, translate=(0.08, 0.08), scale=(0.9, 1.1), shear=(-4.0, 4.0), p=1.0)),
        ("random_crop", RandomCrop(min_scale=0.65, p=1.0)),
        ("mixup", MixUp(alpha=0.4, p=1.0)),
        ("mosaic", Mosaic(output_size=(640, 640), p=1.0)),
    ]

    for name, transform in transforms:
        sample = random.choice(base_samples)
        before = {
            "image": sample["image"].clone(),
            "boxes": sample["boxes"].clone(),
            "labels": sample["labels"].clone(),
        }
        if getattr(transform, "requires_sample_provider", False):
            after = transform(sample, sample_provider=provider)
        else:
            after = transform(sample)
        save_comparison(
            before=before,
            after=after,
            output_path=output_root / f"{name}.png",
            title=name,
        )


if __name__ == "__main__":
    main()
