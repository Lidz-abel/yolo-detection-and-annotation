"""Dataset utilities for the yolov0 training paths."""

import bisect
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, decode_image

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


def count_manifest_samples(manifest_path):
    """Count non-empty jsonl records without loading every sample into memory."""
    manifest_path = Path(manifest_path)
    with manifest_path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _manifest_stem(manifest_path):
    """Return the manifest name used by the packed dataset directory layout."""
    return Path(manifest_path).stem


def resolve_packed_index_path(manifest_path, packing_format="raw", packed_root=None, packed_chunk_size=None):
    """Resolve the packed index path implied by one manifest and config."""
    packing_format = str(packing_format or "raw").lower()
    if packing_format in {"", "raw", "manifest", "jsonl"}:
        return None

    manifest_path = Path(manifest_path)
    if packed_root is None or str(packed_root).strip() == "":
        packed_root = manifest_path.parents[1] / "packed"
    packed_root = Path(packed_root)

    split_dir = packed_root / packing_format / _manifest_stem(manifest_path)
    if packed_chunk_size is not None and int(packed_chunk_size) > 0:
        return split_dir / f"chunk_{int(packed_chunk_size)}" / "index.json"

    candidates = sorted(
        split_dir.glob("chunk_*/index.json"),
        key=lambda path: int(path.parent.name.split("_", 1)[1]) if path.parent.name.startswith("chunk_") else 0,
        reverse=True,
    )
    if candidates:
        return candidates[0]
    return split_dir / "chunk_<missing>" / "index.json"


def validate_packed_index(index_path, expected_format="pt", expected_min_samples=None):
    """Fail fast when a packed dataset is requested but not readable."""
    index_path = Path(index_path)
    if not index_path.exists():
        raise FileNotFoundError(
            f"Packed {expected_format} index not found: {index_path}. "
            "Build it first with DataSet/Unified/pack_detection_chunks.py."
        )

    index = json.loads(index_path.read_text(encoding="utf-8"))
    if str(index.get("format", "")).lower() != str(expected_format).lower():
        raise ValueError(
            f"Packed index format mismatch: expected {expected_format}, got {index.get('format')} at {index_path}"
        )
    chunks = index.get("chunks", [])
    if not chunks:
        raise ValueError(f"Packed index has no chunks: {index_path}")
    first_chunk = index_path.parent / chunks[0]["file"]
    if not first_chunk.exists():
        raise FileNotFoundError(f"Packed chunk referenced by index is missing: {first_chunk}")
    summary = {
        "index_path": str(index_path),
        "format": index["format"],
        "chunk_size": int(index["chunk_size"]),
        "total_samples": int(index["total_samples"]),
        "num_chunks": len(chunks),
    }
    if expected_min_samples is not None and summary["total_samples"] < int(expected_min_samples):
        raise ValueError(
            f"Packed index has too few samples: {summary['total_samples']} < {int(expected_min_samples)} at {index_path}. "
            "Rebuild the packed dataset from the full manifest."
        )
    return summary


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


def _pack_encoded_targets(encoded):
    """Convert one encoded-target tuple into a named target dictionary."""
    (
        target_cls,
        target_box,
        target_obj,
        object_mask,
        noobj_mask,
        collision_count,
        ignored_count,
        dropped_gt_count,
    ) = encoded
    return {
        "target_cls": target_cls,
        "target_box": target_box,
        "target_obj": target_obj,
        "object_mask": object_mask,
        "noobj_mask": noobj_mask,
        "collision_count": collision_count,
        "ignored_count": ignored_count,
        "dropped_gt_count": dropped_gt_count,
    }


class PackedPtStore:
    """Read packed `.pt` chunks produced by DataSet/Unified/pack_detection_chunks.py."""

    def __init__(self, index_path, max_cached_chunks=4):
        self.index_path = Path(index_path)
        self.index = json.loads(self.index_path.read_text(encoding="utf-8"))
        if str(self.index.get("format", "")).lower() != "pt":
            raise ValueError(f"PackedPtStore only supports pt indexes, got {self.index.get('format')}")
        self.max_cached_chunks = max(1, int(max_cached_chunks))
        self.chunk_paths = [
            (self.index_path.parent / chunk_info["file"]).resolve()
            for chunk_info in self.index["chunks"]
        ]
        self.chunk_sizes = [int(chunk_info["sample_count"]) for chunk_info in self.index["chunks"]]
        self.cumulative_sizes = []
        running = 0
        for size in self.chunk_sizes:
            running += size
            self.cumulative_sizes.append(running)
        self._chunk_cache = OrderedDict()

    def __len__(self):
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0

    def _load_chunk(self, chunk_id):
        cached = self._chunk_cache.get(chunk_id)
        if cached is not None:
            self._chunk_cache.move_to_end(chunk_id)
            return cached

        payload = torch.load(self.chunk_paths[chunk_id], map_location="cpu", weights_only=False)
        cached = {
            "sample_count": int(payload["sample_count"][0].item()),
            "image_blob": payload["image_blob"].contiguous().numpy().tobytes(),
            "image_offsets": payload["image_offsets"].tolist(),
            "meta_blob": payload["meta_blob"].contiguous().numpy().tobytes(),
            "meta_offsets": payload["meta_offsets"].tolist(),
            "record_cache": {},
        }
        self._chunk_cache[chunk_id] = cached
        if len(self._chunk_cache) > self.max_cached_chunks:
            self._chunk_cache.popitem(last=False)
        return cached

    @staticmethod
    def _decode_image(image_bytes, image_size):
        byte_tensor = torch.from_numpy(np.frombuffer(image_bytes, dtype=np.uint8).copy())
        image = decode_image(byte_tensor, mode=ImageReadMode.RGB).float() / 255.0
        if image.shape[-2:] != (image_size, image_size):
            image = F.interpolate(
                image.unsqueeze(0),
                size=(image_size, image_size),
                mode="bilinear",
                align_corners=False,
            )[0]
        return image.contiguous()

    def __getitem__(self, index):
        chunk_id = bisect.bisect_right(self.cumulative_sizes, index)
        start = 0 if chunk_id == 0 else self.cumulative_sizes[chunk_id - 1]
        local_index = index - start
        chunk = self._load_chunk(chunk_id)

        record = chunk["record_cache"].get(local_index)
        if record is None:
            meta_offsets = chunk["meta_offsets"]
            meta_start = meta_offsets[local_index]
            meta_end = meta_offsets[local_index + 1]
            record = json.loads(chunk["meta_blob"][meta_start:meta_end].decode("utf-8"))
            chunk["record_cache"][local_index] = record

        image_offsets = chunk["image_offsets"]
        image_start = image_offsets[local_index]
        image_end = image_offsets[local_index + 1]
        return record, chunk["image_blob"][image_start:image_end]


class DetectionDataset(Dataset):
    """Load unified manifest samples and produce images plus detector targets."""

    def __init__(
        self,
        manifest_path,
        image_size=320,
        grid_size=10,
        grid_sizes=None,
        feature_levels=None,
        num_classes=80,
        num_boxes=1,
        anchors=None,
        anchors_by_level=None,
        anchor_positive_iou=0.25,
        anchor_ignore_iou=0.5,
        anchor_match_metric="iou",
        anchor_shape_ratio=4.0,
        anchor_ignore_shape_ratio=None,
        max_samples=None,
        packing_format="raw",
        packed_root=None,
        packed_chunk_size=None,
        packed_cache_size=4,
        require_packed=False,
    ):
        super().__init__()
        self.manifest_path = Path(manifest_path)
        self.image_size = image_size
        self.grid_size = grid_size
        self.grid_sizes = [int(grid_size)] if not grid_sizes else [int(size) for size in grid_sizes]
        self.feature_levels = feature_levels or [f"scale_{index}" for index in range(len(self.grid_sizes))]
        self.multiscale = len(self.grid_sizes) > 1
        self.num_classes = num_classes
        self.num_boxes = num_boxes
        self.anchors = anchors or []
        self.anchors_by_level = anchors_by_level or {}
        self.anchor_positive_iou = anchor_positive_iou
        self.anchor_ignore_iou = anchor_ignore_iou
        self.anchor_match_metric = anchor_match_metric
        self.anchor_shape_ratio = anchor_shape_ratio
        self.anchor_ignore_shape_ratio = anchor_ignore_shape_ratio
        self.packing_format = str(packing_format or "raw").lower()
        self.storage_mode = "raw"
        self.packed_index_path = None
        self.packed_summary = None
        self.packed_store = None
        self.max_samples = int(max_samples) if max_samples is not None and int(max_samples) > 0 else None

        if self.packing_format == "pt":
            self.packed_index_path = resolve_packed_index_path(
                manifest_path=self.manifest_path,
                packing_format=self.packing_format,
                packed_root=packed_root,
                packed_chunk_size=packed_chunk_size,
            )
            expected_min_samples = self.max_samples or count_manifest_samples(self.manifest_path)
            self.packed_summary = validate_packed_index(
                self.packed_index_path,
                expected_format="pt",
                expected_min_samples=expected_min_samples,
            )
            self.packed_store = PackedPtStore(self.packed_index_path, max_cached_chunks=packed_cache_size)
            self.samples = None
            self.storage_mode = "pt"
        else:
            if require_packed:
                raise ValueError(f"require_packed=True but packing_format={self.packing_format!r}")
            self.samples = load_manifest(self.manifest_path)
            if self.max_samples is not None:
                self.samples = self.samples[: self.max_samples]

    def __len__(self):
        if self.storage_mode == "pt":
            length = len(self.packed_store)
            return min(length, self.max_samples) if self.max_samples is not None else length
        return len(self.samples)

    def _load_image(self, image_path):
        """Read an RGB image, resize it, and convert it to CHW float tensor."""
        image = Image.open(image_path).convert("RGB")
        image = image.resize((self.image_size, self.image_size))
        array = np.array(image, dtype=np.float32)
        return torch.from_numpy(array).permute(2, 0, 1) / 255.0

    def _load_sample(self, index):
        """Load one raw or packed sample and return `(record, image_tensor)`."""
        if self.storage_mode == "pt":
            sample, image_bytes = self.packed_store[index]
            return sample, PackedPtStore._decode_image(image_bytes, self.image_size)
        sample = self.samples[index]
        return sample, self._load_image(sample["image_path"])

    def __getitem__(self, index):
        """Return one image and its grid-formatted supervision tensors."""
        sample, image = self._load_sample(index)
        image_path = sample.get("image_path", "")
        orig_width = sample["width"]
        orig_height = sample["height"]

        boxes = torch.tensor(sample["boxes"], dtype=torch.float32)
        labels = torch.tensor(sample["labels"], dtype=torch.long)
        resized_boxes = resize_boxes_xyxy(
            boxes=boxes,
            orig_width=orig_width,
            orig_height=orig_height,
            new_size=self.image_size,
        )

        if self.multiscale:
            multiscale_targets = {}
            for level_name, level_grid_size in zip(self.feature_levels, self.grid_sizes):
                level_anchors = self.anchors_by_level.get(level_name, self.anchors)
                multiscale_targets[level_name] = _pack_encoded_targets(
                    encode_target(
                        boxes=resized_boxes,
                        labels=labels,
                        image_size=self.image_size,
                        grid_size=level_grid_size,
                        num_classes=self.num_classes,
                        num_boxes=self.num_boxes,
                        anchors=level_anchors,
                        anchor_positive_iou=self.anchor_positive_iou,
                        anchor_ignore_iou=self.anchor_ignore_iou,
                        anchor_match_metric=self.anchor_match_metric,
                        anchor_shape_ratio=self.anchor_shape_ratio,
                        anchor_ignore_shape_ratio=self.anchor_ignore_shape_ratio,
                    )
                )
            target = {
                "multiscale_targets": multiscale_targets,
                "boxes": resized_boxes,
                "labels": labels,
                "sample_id": sample["sample_id"],
                "image_path": image_path,
                "dataset_source": sample.get("dataset_source", "unknown"),
                "original_size": torch.tensor([orig_height, orig_width], dtype=torch.float32),
                "resized_size": torch.tensor([self.image_size, self.image_size], dtype=torch.float32),
            }
            return image, target

        target_dict = _pack_encoded_targets(
            encode_target(
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
                anchor_ignore_shape_ratio=self.anchor_ignore_shape_ratio,
            )
        )

        target = {
            **target_dict,
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
    if "multiscale_targets" in batch[0][1]:
        scale_names = list(batch[0][1]["multiscale_targets"].keys())
        multiscale_targets = {}
        for scale_name in scale_names:
            multiscale_targets[scale_name] = {
                key: torch.stack([item[1]["multiscale_targets"][scale_name][key] for item in batch], dim=0)
                for key in (
                    "target_cls",
                    "target_box",
                    "target_obj",
                    "object_mask",
                    "noobj_mask",
                    "collision_count",
                    "ignored_count",
                    "dropped_gt_count",
                )
            }
        targets = {
            "multiscale_targets": multiscale_targets,
            "boxes": [item[1]["boxes"] for item in batch],
            "labels": [item[1]["labels"] for item in batch],
            "sample_id": [item[1]["sample_id"] for item in batch],
            "image_path": [item[1]["image_path"] for item in batch],
            "dataset_source": [item[1]["dataset_source"] for item in batch],
            "original_size": torch.stack([item[1]["original_size"] for item in batch], dim=0),
            "resized_size": torch.stack([item[1]["resized_size"] for item in batch], dim=0),
        }
        return images, targets

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
