import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from data.detection_dataset import DetectionDataset, detection_collate_fn


def parse_args():
    parser = argparse.ArgumentParser(description="Test the real-data DetectionDataset for MiniYOLO.")
    parser.add_argument(
        "--manifest",
        type=str,
        default="/home/lidz/YOLO/DataSet/Unified/manifests/all_val.jsonl",
        help="Path to a unified manifest jsonl file.",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--grid-size", type=int, default=7)
    parser.add_argument("--num-classes", type=int, default=20)
    parser.add_argument("--max-samples", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    return parser.parse_args()


def inspect_single_sample(dataset):
    image, target = dataset[0]

    print("=== Single Sample Check ===")
    print("image shape:", image.shape)
    print("target_cls shape:", target["target_cls"].shape)
    print("target_box shape:", target["target_box"].shape)
    print("object_mask shape:", target["object_mask"].shape)
    print("sample_id:", target["sample_id"])
    print("image_path:", target["image_path"])
    print("original_size:", target["original_size"])
    print("resized_size:", target["resized_size"])
    print("resized boxes:", target["boxes"])
    print("labels:", target["labels"])

    active_indices = torch.nonzero(target["object_mask"])
    print("active grid indices:", active_indices)

    for idx in active_indices:
        gy, gx = idx.tolist()
        print(f"active cell ({gy}, {gx}) class index:", torch.argmax(target["target_cls"][gy, gx]).item())
        print(f"active cell ({gy}, {gx}) target box:", target["target_box"][gy, gx])


def inspect_batch(dataset, batch_size):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=detection_collate_fn,
    )

    images, targets = next(iter(loader))

    print("\n=== Batch Check ===")
    print("images shape:", images.shape)
    print("batch target_cls shape:", targets["target_cls"].shape)
    print("batch target_box shape:", targets["target_box"].shape)
    print("batch object_mask shape:", targets["object_mask"].shape)
    print("batch sample_id:", targets["sample_id"])
    print("batch original_size shape:", targets["original_size"].shape)
    print("batch resized_size shape:", targets["resized_size"].shape)


def main():
    args = parse_args()

    dataset = DetectionDataset(
        manifest_path=args.manifest,
        image_size=args.image_size,
        grid_size=args.grid_size,
        num_classes=args.num_classes,
        max_samples=args.max_samples,
    )

    print("dataset length:", len(dataset))
    inspect_single_sample(dataset)
    inspect_batch(dataset, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
