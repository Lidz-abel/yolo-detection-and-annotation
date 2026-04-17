from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, read_image

from detection_augmentations import default_train_augmentation_pipeline, sanitize_boxes


def load_manifest(manifest_path: str | Path, max_samples: int | None = None) -> list[dict[str, Any]]:
    manifest_path = Path(manifest_path)
    records: list[dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            records.append(__import__("json").loads(line))
            if max_samples is not None and len(records) >= max_samples:
                break
    return records


class AugmentedDetectionDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        max_samples: int | None = None,
        image_mode: str = "rgb",
        augment: Any | None = None,
        enable_mixup_mosaic: bool = True,
    ) -> None:
        self.records = load_manifest(manifest_path, max_samples=max_samples)
        self.image_mode = image_mode
        self.augment = augment
        self.enable_mixup_mosaic = enable_mixup_mosaic

    def __len__(self) -> int:
        return len(self.records)

    def _read_image(self, image_path: str) -> torch.Tensor:
        read_mode = ImageReadMode.RGB if self.image_mode.lower() == "rgb" else ImageReadMode.UNCHANGED
        return read_image(image_path, mode=read_mode)

    def _build_sample(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image = self._read_image(record["image_path"])
        boxes = torch.tensor(record["boxes"], dtype=torch.float32).reshape(-1, 4)
        labels = torch.tensor(record["labels"], dtype=torch.long)
        boxes, labels = sanitize_boxes(boxes, labels, int(record["height"]), int(record["width"]))
        return {
            "sample_id": record["sample_id"],
            "image": image,
            "boxes": boxes,
            "labels": labels,
            "meta": {
                "dataset_source": record["dataset_source"],
                "split": record["split"],
                "image_id": record["image_id"],
                "image_path": record["image_path"],
                "width": record["width"],
                "height": record["height"],
            },
        }

    def _sample_provider(self) -> dict[str, Any]:
        import random

        random_index = random.randrange(len(self.records))
        return self._build_sample(random_index)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self._build_sample(index)
        if self.augment is not None:
            provider = self._sample_provider if self.enable_mixup_mosaic else None
            sample = self.augment(sample, sample_provider=provider)

        target = {
            "boxes": sample["boxes"],
            "labels": sample["labels"],
            "image_id": sample["meta"]["image_id"],
        }
        return {
            "sample_id": sample["sample_id"],
            "image": sample["image"],
            "target": target,
            "meta": sample["meta"],
        }


def build_default_augmented_dataset(
    manifest_path: str | Path,
    max_samples: int | None = None,
    image_mode: str = "rgb",
) -> AugmentedDetectionDataset:
    return AugmentedDetectionDataset(
        manifest_path=manifest_path,
        max_samples=max_samples,
        image_mode=image_mode,
        augment=default_train_augmentation_pipeline(),
        enable_mixup_mosaic=True,
    )
