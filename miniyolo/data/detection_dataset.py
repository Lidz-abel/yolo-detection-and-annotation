import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from data.target_encoder import encode_target


def load_manifest(manifest_path):
    manifest_path = Path(manifest_path)
    samples = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    return samples


def resize_boxes_xyxy(boxes, orig_width, orig_height, new_size):
    if boxes.numel() == 0:
        return boxes.clone()

    scale_x = new_size / float(orig_width)
    scale_y = new_size / float(orig_height)

    resized_boxes = boxes.clone().float()
    resized_boxes[:, 0] = resized_boxes[:, 0] * scale_x
    resized_boxes[:, 2] = resized_boxes[:, 2] * scale_x
    resized_boxes[:, 1] = resized_boxes[:, 1] * scale_y
    resized_boxes[:, 3] = resized_boxes[:, 3] * scale_y
    return resized_boxes


class DetectionDataset(Dataset):
    def __init__(
        self,
        manifest_path,
        image_size=224,
        grid_size=7,
        num_classes=20,
        max_samples=None,
    ):
        super().__init__()
        self.manifest_path = Path(manifest_path)
        self.image_size = image_size
        self.grid_size = grid_size
        self.num_classes = num_classes

        self.samples = load_manifest(self.manifest_path)
        if max_samples is not None:
            self.samples = self.samples[:max_samples]

    def __len__(self):
        return len(self.samples)

    def _load_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image = image.resize((self.image_size, self.image_size))

        image_tensor = torch.from_numpy(
            __import__("numpy").array(image, dtype="float32")
        ).permute(2, 0, 1) / 255.0
        return image_tensor

    def __getitem__(self, index):
        sample = self.samples[index]

        image_path = sample["image_path"]
        orig_width = sample["width"]
        orig_height = sample["height"]

        image = self._load_image(image_path)

        boxes = torch.tensor(sample["boxes"], dtype=torch.float32)
        labels = torch.tensor(sample["labels"], dtype=torch.long)
        resized_boxes = resize_boxes_xyxy(
            boxes=boxes,
            orig_width=orig_width,
            orig_height=orig_height,
            new_size=self.image_size,
        )

        target_cls, target_box, object_mask = encode_target(
            boxes=resized_boxes,
            labels=labels,
            image_size=self.image_size,
            grid_size=self.grid_size,
            num_classes=self.num_classes,
        )

        target = {
            "target_cls": target_cls,
            "target_box": target_box,
            "object_mask": object_mask,
            "boxes": resized_boxes,
            "labels": labels,
            "sample_id": sample["sample_id"],
            "image_path": image_path,
            "original_size": torch.tensor([orig_height, orig_width], dtype=torch.float32),
            "resized_size": torch.tensor([self.image_size, self.image_size], dtype=torch.float32),
        }

        return image, target


def detection_collate_fn(batch):
    images = torch.stack([item[0] for item in batch], dim=0)
    target_cls = torch.stack([item[1]["target_cls"] for item in batch], dim=0)
    target_box = torch.stack([item[1]["target_box"] for item in batch], dim=0)
    object_mask = torch.stack([item[1]["object_mask"] for item in batch], dim=0)

    targets = {
        "target_cls": target_cls,
        "target_box": target_box,
        "object_mask": object_mask,
        "boxes": [item[1]["boxes"] for item in batch],
        "labels": [item[1]["labels"] for item in batch],
        "sample_id": [item[1]["sample_id"] for item in batch],
        "image_path": [item[1]["image_path"] for item in batch],
        "original_size": torch.stack([item[1]["original_size"] for item in batch], dim=0),
        "resized_size": torch.stack([item[1]["resized_size"] for item in batch], dim=0),
    }
    return images, targets
