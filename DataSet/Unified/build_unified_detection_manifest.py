#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path


VOC_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a unified object-detection manifest for VOC2012 and COCO2017."
    )
    parser.add_argument(
        "--voc-root",
        type=Path,
        default=Path(
            "/home/lidz/YOLO/DataSet/VOC2012/VOC12/OpenDataLab___PASCAL_VOC2012/VOCdevkit/VOC2012"
        ),
    )
    parser.add_argument(
        "--coco-root",
        type=Path,
        default=Path("/home/lidz/YOLO/DataSet/COCO2017/OpenDataLab___COCO_2017"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/home/lidz/YOLO/DataSet/Unified"),
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_voc_records(voc_root: Path) -> tuple[dict, dict[str, list[dict]]]:
    class_to_id = {name: idx for idx, name in enumerate(VOC_CLASSES)}
    image_sets_dir = voc_root / "ImageSets" / "Main"
    annotations_dir = voc_root / "Annotations"
    images_dir = voc_root / "JPEGImages"

    records_by_split: dict[str, list[dict]] = {"train": [], "val": []}

    for split in ("train", "val"):
        split_ids = (image_sets_dir / f"{split}.txt").read_text(encoding="utf-8").splitlines()
        for image_id in split_ids:
            xml_path = annotations_dir / f"{image_id}.xml"
            image_path = images_dir / f"{image_id}.jpg"
            root = ET.parse(xml_path).getroot()

            width = int(root.findtext("size/width"))
            height = int(root.findtext("size/height"))
            boxes = []
            labels = []
            label_names = []
            original_category_ids = []
            iscrowd = []
            difficult = []
            truncated = []
            occluded = []
            areas = []

            for obj in root.findall("object"):
                name = obj.findtext("name")
                bbox = obj.find("bndbox")
                x1 = float(bbox.findtext("xmin"))
                y1 = float(bbox.findtext("ymin"))
                x2 = float(bbox.findtext("xmax"))
                y2 = float(bbox.findtext("ymax"))

                boxes.append([x1, y1, x2, y2])
                labels.append(class_to_id[name])
                label_names.append(name)
                original_category_ids.append(class_to_id[name])
                iscrowd.append(0)
                difficult.append(int(obj.findtext("difficult", default="0")))
                truncated.append(int(obj.findtext("truncated", default="0")))
                occluded.append(int(obj.findtext("occluded", default="0")))
                areas.append((x2 - x1) * (y2 - y1))

            record = {
                "sample_id": f"voc2012_{split}_{image_id}",
                "dataset_source": "voc2012",
                "split": split,
                "image_id": image_id,
                "image_path": str(image_path.resolve()),
                "image_rel_path": str(image_path.relative_to(voc_root)),
                "width": width,
                "height": height,
                "boxes": boxes,
                "labels": labels,
                "label_names": label_names,
                "original_category_ids": original_category_ids,
                "iscrowd": iscrowd,
                "difficult": difficult,
                "truncated": truncated,
                "occluded": occluded,
                "areas": areas,
                "annotation_count": len(boxes),
            }
            records_by_split[split].append(record)

    metadata = {
        "dataset_source": "voc2012",
        "dataset_root": str(voc_root.resolve()),
        "image_dir": str(images_dir.resolve()),
        "annotation_dir": str(annotations_dir.resolve()),
        "class_to_id": class_to_id,
        "id_to_class": {str(v): k for k, v in class_to_id.items()},
        "splits": {split: len(records) for split, records in records_by_split.items()},
    }
    return metadata, records_by_split


def build_coco_records(coco_root: Path) -> tuple[dict, dict[str, list[dict]]]:
    annotations_dir = coco_root / "annotations"
    records_by_split: dict[str, list[dict]] = {"train": [], "val": []}
    original_category_name_map = {}
    contiguous_category_name_map = {}

    for split in ("train", "val"):
        annotation_path = annotations_dir / f"instances_{split}2017.json"
        payload = json.loads(annotation_path.read_text(encoding="utf-8"))

        categories = sorted(payload["categories"], key=lambda item: item["id"])
        original_to_contiguous = {category["id"]: idx for idx, category in enumerate(categories)}
        category_names = {category["id"]: category["name"] for category in categories}

        original_category_name_map[split] = {str(cat["id"]): cat["name"] for cat in categories}
        contiguous_category_name_map[split] = {
            str(original_to_contiguous[cat["id"]]): cat["name"] for cat in categories
        }

        annotations_by_image: dict[int, list[dict]] = defaultdict(list)
        for annotation in payload["annotations"]:
            annotations_by_image[annotation["image_id"]].append(annotation)

        split_image_dir = coco_root / f"{split}2017"
        for image in payload["images"]:
            anns = annotations_by_image.get(image["id"], [])

            boxes = []
            labels = []
            label_names = []
            original_category_ids = []
            iscrowd = []
            difficult = []
            truncated = []
            occluded = []
            areas = []

            for ann in anns:
                x, y, w, h = ann["bbox"]
                x1 = float(x)
                y1 = float(y)
                x2 = float(x + w)
                y2 = float(y + h)
                original_category_id = ann["category_id"]

                boxes.append([x1, y1, x2, y2])
                labels.append(original_to_contiguous[original_category_id])
                label_names.append(category_names[original_category_id])
                original_category_ids.append(original_category_id)
                iscrowd.append(int(ann.get("iscrowd", 0)))
                difficult.append(0)
                truncated.append(0)
                occluded.append(0)
                areas.append(float(ann.get("area", w * h)))

            image_id = f"{image['id']:012d}"
            image_path = split_image_dir / image["file_name"]
            record = {
                "sample_id": f"coco2017_{split}_{image_id}",
                "dataset_source": "coco2017",
                "split": split,
                "image_id": image_id,
                "image_path": str(image_path.resolve()),
                "image_rel_path": str(image_path.relative_to(coco_root)),
                "width": int(image["width"]),
                "height": int(image["height"]),
                "boxes": boxes,
                "labels": labels,
                "label_names": label_names,
                "original_category_ids": original_category_ids,
                "iscrowd": iscrowd,
                "difficult": difficult,
                "truncated": truncated,
                "occluded": occluded,
                "areas": areas,
                "annotation_count": len(boxes),
            }
            records_by_split[split].append(record)

    metadata = {
        "dataset_source": "coco2017",
        "dataset_root": str(coco_root.resolve()),
        "annotation_dir": str(annotations_dir.resolve()),
        "splits": {split: len(records) for split, records in records_by_split.items()},
        "original_category_id_to_name_by_split": original_category_name_map,
        "contiguous_category_id_to_name_by_split": contiguous_category_name_map,
    }
    return metadata, records_by_split


def build_schema() -> dict:
    return {
        "record_type": "object_detection_sample",
        "box_format": "xyxy",
        "label_policy": {
            "labels": "dataset-local contiguous ids starting from 0",
            "original_category_ids": "original ids from the source dataset",
        },
        "fields": {
            "sample_id": "global unique sample identifier",
            "dataset_source": "voc2012 or coco2017",
            "split": "train or val",
            "image_id": "original image id",
            "image_path": "absolute path to the image file",
            "image_rel_path": "path relative to dataset root",
            "width": "image width in pixels",
            "height": "image height in pixels",
            "boxes": "list of bounding boxes in [x1, y1, x2, y2]",
            "labels": "contiguous dataset-local class ids",
            "label_names": "class names aligned with labels",
            "original_category_ids": "original source-dataset class ids",
            "iscrowd": "crowd flag for each annotation",
            "difficult": "VOC difficult flag, COCO default 0",
            "truncated": "VOC truncated flag, COCO default 0",
            "occluded": "VOC occluded flag when present, COCO default 0",
            "areas": "annotation areas",
            "annotation_count": "number of boxes in the sample",
        },
    }


def summarize(records_by_name: dict[str, list[dict]]) -> dict:
    summary = {}
    for name, records in records_by_name.items():
        total_boxes = sum(record["annotation_count"] for record in records)
        empty_images = sum(1 for record in records if record["annotation_count"] == 0)
        summary[name] = {
            "images": len(records),
            "boxes": total_boxes,
            "empty_images": empty_images,
        }
    return summary


def main() -> None:
    args = parse_args()

    manifests_dir = args.output_root / "manifests"
    metadata_dir = args.output_root / "metadata"
    ensure_dir(args.output_root)
    ensure_dir(manifests_dir)
    ensure_dir(metadata_dir)

    voc_metadata, voc_records = build_voc_records(args.voc_root)
    coco_metadata, coco_records = build_coco_records(args.coco_root)

    all_manifests = {
        "voc2012_train": voc_records["train"],
        "voc2012_val": voc_records["val"],
        "coco2017_train": coco_records["train"],
        "coco2017_val": coco_records["val"],
    }
    all_manifests["all_train"] = voc_records["train"] + coco_records["train"]
    all_manifests["all_val"] = voc_records["val"] + coco_records["val"]
    all_manifests["all"] = all_manifests["all_train"] + all_manifests["all_val"]

    for name, records in all_manifests.items():
        write_jsonl(manifests_dir / f"{name}.jsonl", records)

    write_json(metadata_dir / "schema.json", build_schema())
    write_json(
        metadata_dir / "class_maps.json",
        {
            "voc2012": {
                "class_to_id": voc_metadata["class_to_id"],
                "id_to_class": voc_metadata["id_to_class"],
            },
            "coco2017": {
                "original_category_id_to_name_by_split": coco_metadata[
                    "original_category_id_to_name_by_split"
                ],
                "contiguous_category_id_to_name_by_split": coco_metadata[
                    "contiguous_category_id_to_name_by_split"
                ],
            },
        },
    )
    write_json(
        metadata_dir / "summary.json",
        {
            "voc2012": voc_metadata,
            "coco2017": coco_metadata,
            "manifests": summarize(all_manifests),
        },
    )

    print(f"Wrote manifests to {manifests_dir}")
    print(f"Wrote metadata to {metadata_dir}")
    for name, info in summarize(all_manifests).items():
        print(f"{name}: {info['images']} images, {info['boxes']} boxes, {info['empty_images']} empty")


if __name__ == "__main__":
    main()
