"""Dataset utilities for the yolov0 training paths."""

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from data.target_encoder import encode_target


def load_manifest(manifest_path):
    """Read unified jsonl manifests into an in-memory sample list."""
    manifest_path = Path(manifest_path)
    samples = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    return samples


def resize_boxes_xyxy(boxes, orig_width, orig_height, new_size):
    """Resize xyxy boxes together with the resized training image."""
    if boxes.numel() == 0:
        return boxes.clone()

    scale_x = new_size / float(orig_width)
    scale_y = new_size / float(orig_height)

    resized_boxes = boxes.clone().float()
    resized_boxes[:, 0] *= scale_x
    resized_boxes[:, 2] *= scale_x
    resized_boxes[:, 1] *= scale_y
    resized_boxes[:, 3] *= scale_y
    return resized_boxes


class DetectionDataset(Dataset):
    """Load unified manifest samples and produce images plus detector targets."""

    def __init__(
        self,
        manifest_path,
        image_size=320,
        grid_size=10,
        num_classes=80,
        num_boxes=1,
        anchors=None,
        anchor_positive_iou=0.25,
        anchor_ignore_iou=0.5,
        anchor_match_metric="iou",
        anchor_shape_ratio=4.0,
        max_samples=None,
    ):
        super().__init__()
        self.manifest_path = Path(manifest_path)
        self.image_size = image_size
        self.grid_size = grid_size
        self.num_classes = num_classes
        self.num_boxes = num_boxes
        self.anchors = anchors or []
        self.anchor_positive_iou = anchor_positive_iou
        self.anchor_ignore_iou = anchor_ignore_iou
        self.anchor_match_metric = anchor_match_metric
        self.anchor_shape_ratio = anchor_shape_ratio

        self.samples = load_manifest(self.manifest_path)
        # A non-positive max_samples means "use the full manifest".
        if max_samples is not None and int(max_samples) > 0:
            self.samples = self.samples[:max_samples]

    def __len__(self):
        return len(self.samples)

    def _load_image(self, image_path):
        """Read an RGB image, resize it, and convert it to CHW float tensor."""
        image = Image.open(image_path).convert("RGB")
        image = image.resize((self.image_size, self.image_size))
        array = np.array(image, dtype=np.float32)
        return torch.from_numpy(array).permute(2, 0, 1) / 255.0

    def __getitem__(self, index):
        """Return one image and its grid-formatted supervision tensors."""
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

        (
            target_cls,
            target_box,
            target_obj,
            object_mask,
            noobj_mask,
            collision_count,
            ignored_count,
            dropped_gt_count,
        ) = encode_target(
            boxes=resized_boxes,
            labels=labels,
            image_size=self.image_size,
            grid_size=self.grid_size,
            num_classes=self.num_classes,
            num_boxes=self.num_boxes,
            anchors=self.anchors,
            anchor_positive_iou=self.anchor_positive_iou,
            anchor_ignore_iou=self.anchor_ignore_iou,
            anchor_match_metric=self.anchor_match_metric,
            anchor_shape_ratio=self.anchor_shape_ratio,
        )

        target = {
            "target_cls": target_cls,
            "target_box": target_box,
            "target_obj": target_obj,
            "object_mask": object_mask,
            "noobj_mask": noobj_mask,
            "collision_count": collision_count,
            "ignored_count": ignored_count,
            "dropped_gt_count": dropped_gt_count,
            "boxes": resized_boxes,
            "labels": labels,
            "sample_id": sample["sample_id"],
            "image_path": image_path,
            "dataset_source": sample.get("dataset_source", "unknown"),
            "original_size": torch.tensor([orig_height, orig_width], dtype=torch.float32),
            "resized_size": torch.tensor([self.image_size, self.image_size], dtype=torch.float32),
        }
        return image, target


def detection_collate_fn(batch):
    """Stack fixed-size tensors and keep raw annotations as per-sample lists."""
    images = torch.stack([item[0] for item in batch], dim=0)
    target_cls = torch.stack([item[1]["target_cls"] for item in batch], dim=0)
    target_box = torch.stack([item[1]["target_box"] for item in batch], dim=0)
    target_obj = torch.stack([item[1]["target_obj"] for item in batch], dim=0)
    object_mask = torch.stack([item[1]["object_mask"] for item in batch], dim=0)
    noobj_mask = torch.stack([item[1]["noobj_mask"] for item in batch], dim=0)
    collision_count = torch.stack([item[1]["collision_count"] for item in batch], dim=0)
    ignored_count = torch.stack([item[1]["ignored_count"] for item in batch], dim=0)
    dropped_gt_count = torch.stack([item[1]["dropped_gt_count"] for item in batch], dim=0)

    targets = {
        "target_cls": target_cls,
        "target_box": target_box,
        "target_obj": target_obj,
        "object_mask": object_mask,
        "noobj_mask": noobj_mask,
        "collision_count": collision_count,
        "ignored_count": ignored_count,
        "dropped_gt_count": dropped_gt_count,
        "boxes": [item[1]["boxes"] for item in batch],
        "labels": [item[1]["labels"] for item in batch],
        "sample_id": [item[1]["sample_id"] for item in batch],
        "image_path": [item[1]["image_path"] for item in batch],
        "dataset_source": [item[1]["dataset_source"] for item in batch],
        "original_size": torch.stack([item[1]["original_size"] for item in batch], dim=0),
        "resized_size": torch.stack([item[1]["resized_size"] for item in batch], dim=0),
    }
    return images, targets
