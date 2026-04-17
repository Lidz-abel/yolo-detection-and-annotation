#!/usr/bin/env python3
from __future__ import annotations

import bisect
import io
import json
import math
import pickle
from pathlib import Path
from typing import Any

import h5py
import msgpack
import numpy as np
import torch
from safetensors.torch import load_file as load_safetensors_file
from safetensors.torch import save_file as save_safetensors_file
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, decode_image, read_image


FORMAT_EXTENSIONS = {
    "pickle": ".pkl",
    "npz": ".npz",
    "pt": ".pt",
    "safetensors": ".safetensors",
    "hdf5": ".h5",
    "msgpack": ".msgpack",
}


SUPPORTED_FORMATS = tuple(FORMAT_EXTENSIONS.keys())


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


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


def sample_metadata(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in record.items()
        if key not in {"image_path", "image_rel_path"}
    }


def build_chunk_payload(records: list[dict[str, Any]]) -> dict[str, Any]:
    image_offsets = [0]
    meta_offsets = [0]
    image_parts: list[bytes] = []
    meta_parts: list[bytes] = []

    for record in records:
        image_bytes = Path(record["image_path"]).read_bytes()
        image_parts.append(image_bytes)
        image_offsets.append(image_offsets[-1] + len(image_bytes))

        meta_bytes = json.dumps(sample_metadata(record), ensure_ascii=False).encode("utf-8")
        meta_parts.append(meta_bytes)
        meta_offsets.append(meta_offsets[-1] + len(meta_bytes))

    return {
        "sample_count": len(records),
        "image_blob": b"".join(image_parts),
        "image_offsets": image_offsets,
        "meta_blob": b"".join(meta_parts),
        "meta_offsets": meta_offsets,
    }


def _bytes_to_numpy(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype=np.uint8).copy()


def _numpy_to_bytes(data: np.ndarray) -> bytes:
    return data.astype(np.uint8, copy=False).tobytes()


def _chunk_to_common(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_count": int(payload["sample_count"]),
        "image_blob": payload["image_blob"],
        "image_offsets": [int(x) for x in payload["image_offsets"]],
        "meta_blob": payload["meta_blob"],
        "meta_offsets": [int(x) for x in payload["meta_offsets"]],
    }


def save_chunk(path: str | Path, payload: dict[str, Any], fmt: str) -> None:
    path = Path(path)

    if fmt == "pickle":
        with path.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return

    if fmt == "npz":
        np.savez_compressed(
            path,
            sample_count=np.array([payload["sample_count"]], dtype=np.int64),
            image_blob=_bytes_to_numpy(payload["image_blob"]),
            image_offsets=np.array(payload["image_offsets"], dtype=np.int64),
            meta_blob=_bytes_to_numpy(payload["meta_blob"]),
            meta_offsets=np.array(payload["meta_offsets"], dtype=np.int64),
        )
        return

    if fmt == "pt":
        torch.save(
            {
                "sample_count": torch.tensor([payload["sample_count"]], dtype=torch.int64),
                "image_blob": torch.from_numpy(_bytes_to_numpy(payload["image_blob"])),
                "image_offsets": torch.tensor(payload["image_offsets"], dtype=torch.int64),
                "meta_blob": torch.from_numpy(_bytes_to_numpy(payload["meta_blob"])),
                "meta_offsets": torch.tensor(payload["meta_offsets"], dtype=torch.int64),
            },
            path,
        )
        return

    if fmt == "safetensors":
        save_safetensors_file(
            {
                "sample_count": torch.tensor([payload["sample_count"]], dtype=torch.int64),
                "image_blob": torch.from_numpy(_bytes_to_numpy(payload["image_blob"])),
                "image_offsets": torch.tensor(payload["image_offsets"], dtype=torch.int64),
                "meta_blob": torch.from_numpy(_bytes_to_numpy(payload["meta_blob"])),
                "meta_offsets": torch.tensor(payload["meta_offsets"], dtype=torch.int64),
            },
            str(path),
        )
        return

    if fmt == "hdf5":
        with h5py.File(path, "w") as handle:
            handle.create_dataset("sample_count", data=np.array([payload["sample_count"]], dtype=np.int64))
            handle.create_dataset("image_blob", data=_bytes_to_numpy(payload["image_blob"]))
            handle.create_dataset("image_offsets", data=np.array(payload["image_offsets"], dtype=np.int64))
            handle.create_dataset("meta_blob", data=_bytes_to_numpy(payload["meta_blob"]))
            handle.create_dataset("meta_offsets", data=np.array(payload["meta_offsets"], dtype=np.int64))
        return

    if fmt == "msgpack":
        packed = msgpack.packb(
            {
                "sample_count": payload["sample_count"],
                "image_blob": payload["image_blob"],
                "image_offsets": payload["image_offsets"],
                "meta_blob": payload["meta_blob"],
                "meta_offsets": payload["meta_offsets"],
            },
            use_bin_type=True,
        )
        path.write_bytes(packed)
        return

    raise ValueError(f"Unsupported format: {fmt}")


def load_chunk(path: str | Path, fmt: str) -> dict[str, Any]:
    path = Path(path)

    if fmt == "pickle":
        with path.open("rb") as handle:
            return _chunk_to_common(pickle.load(handle))

    if fmt == "npz":
        with np.load(path, allow_pickle=False) as payload:
            return {
                "sample_count": int(payload["sample_count"][0]),
                "image_blob": _numpy_to_bytes(payload["image_blob"]),
                "image_offsets": payload["image_offsets"].astype(np.int64).tolist(),
                "meta_blob": _numpy_to_bytes(payload["meta_blob"]),
                "meta_offsets": payload["meta_offsets"].astype(np.int64).tolist(),
            }

    if fmt == "pt":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        return {
            "sample_count": int(payload["sample_count"][0].item()),
            "image_blob": bytes(payload["image_blob"].tolist()),
            "image_offsets": payload["image_offsets"].tolist(),
            "meta_blob": bytes(payload["meta_blob"].tolist()),
            "meta_offsets": payload["meta_offsets"].tolist(),
        }

    if fmt == "safetensors":
        payload = load_safetensors_file(str(path))
        return {
            "sample_count": int(payload["sample_count"][0].item()),
            "image_blob": bytes(payload["image_blob"].tolist()),
            "image_offsets": payload["image_offsets"].tolist(),
            "meta_blob": bytes(payload["meta_blob"].tolist()),
            "meta_offsets": payload["meta_offsets"].tolist(),
        }

    if fmt == "hdf5":
        with h5py.File(path, "r") as handle:
            return {
                "sample_count": int(handle["sample_count"][0]),
                "image_blob": _numpy_to_bytes(handle["image_blob"][:]),
                "image_offsets": handle["image_offsets"][:].astype(np.int64).tolist(),
                "meta_blob": _numpy_to_bytes(handle["meta_blob"][:]),
                "meta_offsets": handle["meta_offsets"][:].astype(np.int64).tolist(),
            }

    if fmt == "msgpack":
        payload = msgpack.unpackb(path.read_bytes(), raw=False)
        return _chunk_to_common(payload)

    raise ValueError(f"Unsupported format: {fmt}")


def decode_image_bytes(image_bytes: bytes, mode: str = "rgb") -> torch.Tensor:
    byte_tensor = torch.from_numpy(np.frombuffer(image_bytes, dtype=np.uint8).copy())
    read_mode = ImageReadMode.RGB if mode.lower() == "rgb" else ImageReadMode.UNCHANGED
    return decode_image(byte_tensor, mode=read_mode)


def detection_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "images": [sample["image"] for sample in batch],
        "targets": [sample["target"] for sample in batch],
        "sample_ids": [sample["sample_id"] for sample in batch],
    }


def move_batch_to_device(batch: dict[str, Any], device: str) -> dict[str, Any]:
    batch["images"] = [image.to(device, non_blocking=True) for image in batch["images"]]
    return batch


class RawManifestDataset(Dataset):
    def __init__(self, manifest_path: str | Path, max_samples: int | None = None, image_mode: str = "rgb"):
        self.records = load_manifest(manifest_path, max_samples=max_samples)
        self.image_mode = image_mode

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        read_mode = ImageReadMode.RGB if self.image_mode.lower() == "rgb" else ImageReadMode.UNCHANGED
        image = read_image(record["image_path"], mode=read_mode)
        return {
            "sample_id": record["sample_id"],
            "image": image,
            "target": sample_metadata(record),
        }


class ChunkedPackedDataset(Dataset):
    def __init__(self, index_path: str | Path, image_mode: str = "rgb"):
        self.index_path = Path(index_path)
        self.index = json.loads(self.index_path.read_text(encoding="utf-8"))
        self.image_mode = image_mode
        self.format_name = self.index["format"]
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

        self._cached_chunk_id: int | None = None
        self._cached_records: list[dict[str, Any]] | None = None
        self._cached_images: list[bytes] | None = None

    def __len__(self) -> int:
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0

    def _load_chunk(self, chunk_id: int) -> None:
        payload = load_chunk(self.chunk_paths[chunk_id], self.format_name)
        meta_blob = payload["meta_blob"]
        meta_offsets = payload["meta_offsets"]
        image_blob = payload["image_blob"]
        image_offsets = payload["image_offsets"]

        records = []
        images = []
        for idx in range(payload["sample_count"]):
            meta_start = meta_offsets[idx]
            meta_end = meta_offsets[idx + 1]
            image_start = image_offsets[idx]
            image_end = image_offsets[idx + 1]
            records.append(json.loads(meta_blob[meta_start:meta_end].decode("utf-8")))
            images.append(image_blob[image_start:image_end])

        self._cached_chunk_id = chunk_id
        self._cached_records = records
        self._cached_images = images

    def __getitem__(self, index: int) -> dict[str, Any]:
        chunk_id = bisect.bisect_right(self.cumulative_sizes, index)
        start = 0 if chunk_id == 0 else self.cumulative_sizes[chunk_id - 1]
        local_index = index - start

        if self._cached_chunk_id != chunk_id:
            self._load_chunk(chunk_id)

        assert self._cached_records is not None
        assert self._cached_images is not None

        record = self._cached_records[local_index]
        image = decode_image_bytes(self._cached_images[local_index], mode=self.image_mode)
        return {
            "sample_id": record["sample_id"],
            "image": image,
            "target": record,
        }


def build_index_payload(
    fmt: str,
    manifest_path: str | Path,
    chunk_size: int,
    max_samples: int | None,
    chunk_files: list[Path],
    chunk_sample_counts: list[int],
) -> dict[str, Any]:
    return {
        "format": fmt,
        "manifest_path": str(Path(manifest_path).resolve()),
        "chunk_size": int(chunk_size),
        "max_samples": max_samples,
        "total_samples": int(sum(chunk_sample_counts)),
        "chunks": [
            {"file": chunk_file.name, "sample_count": int(sample_count)}
            for chunk_file, sample_count in zip(chunk_files, chunk_sample_counts)
        ],
    }


def chunk_records(records: list[dict[str, Any]], chunk_size: int) -> list[list[dict[str, Any]]]:
    total_chunks = math.ceil(len(records) / chunk_size)
    return [records[idx * chunk_size : (idx + 1) * chunk_size] for idx in range(total_chunks)]
