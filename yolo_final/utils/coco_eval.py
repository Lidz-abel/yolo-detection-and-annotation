"""COCO-style evaluation helpers for the COCO subset of the unified manifests."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from utils.prediction import decode_predictions_for_image, select_prediction_for_image


def load_coco_category_info(metadata_path: Path, split: str = "val") -> tuple[dict[int, int], list[dict]]:
    """Load contiguous-to-original category mappings for the COCO subset."""
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    coco_info = payload["coco2017"]
    contiguous_to_name = coco_info["contiguous_category_id_to_name_by_split"][split]
    original_to_name = coco_info["original_category_id_to_name_by_split"][split]
    name_to_original = {name: int(category_id) for category_id, name in original_to_name.items()}

    contiguous_to_original: dict[int, int] = {}
    categories = []
    for contiguous_id_str, name in contiguous_to_name.items():
        contiguous_id = int(contiguous_id_str)
        original_id = name_to_original[name]
        contiguous_to_original[contiguous_id] = original_id
        categories.append({"id": original_id, "name": name})
    categories.sort(key=lambda item: item["id"])
    return contiguous_to_original, categories


def build_coco_gt_dict(manifest_path: Path, categories: list[dict], max_samples: int = 0) -> dict:
    """Convert the COCO manifest subset into one pycocotools-style GT dictionary."""
    images = []
    annotations = []
    annotation_id = 1

    with manifest_path.open("r", encoding="utf-8") as handle:
        for sample_index, line in enumerate(handle, start=1):
            if max_samples > 0 and sample_index > max_samples:
                break
            sample = json.loads(line)
            image_id = int(sample["image_id"])
            images.append(
                {
                    "id": image_id,
                    "width": int(sample["width"]),
                    "height": int(sample["height"]),
                    "file_name": Path(sample["image_path"]).name,
                }
            )
            for box, category_id, area, iscrowd in zip(
                sample["boxes"],
                sample["original_category_ids"],
                sample.get("areas", [0.0] * len(sample["boxes"])),
                sample.get("iscrowd", [0] * len(sample["boxes"])),
            ):
                x1, y1, x2, y2 = box
                annotations.append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": int(category_id),
                        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                        "area": float(area),
                        "iscrowd": int(iscrowd),
                    }
                )
                annotation_id += 1

    return {
        "images": images,
        "annotations": annotations,
        "categories": categories,
        "info": {"description": "YOLOv0 unified COCO subset evaluation"},
        "licenses": [],
    }


def evaluate_coco_subset(
    model,
    dataset,
    device,
    metadata_path: Path,
    manifest_path: Path,
    image_size: int,
    num_classes: int,
    num_boxes: int = 1,
    anchors: list[tuple[float, float]] | dict[str, list[tuple[float, float]]] | None = None,
    box_parameterization: str = "legacy",
    max_samples: int = 0,
    score_threshold: float = 0.05,
    top_k: int = 100,
    nms_iou_threshold: float = 0.5,
    score_alpha: float = 1.0,
    score_beta: float = 1.0,
) -> dict[str, float]:
    """Evaluate one detector on the COCO subset using pycocotools."""
    contiguous_to_original, categories = load_coco_category_info(metadata_path, split="val")
    gt_dict = build_coco_gt_dict(manifest_path, categories, max_samples=max_samples)

    gt_api = COCO()
    gt_api.dataset = gt_dict
    gt_api.createIndex()

    was_training = model.training
    model.eval()

    results = []
    num_samples = len(dataset) if max_samples <= 0 else min(int(max_samples), len(dataset))
    with torch.no_grad():
        for sample_index in range(num_samples):
            image_tensor, target = dataset[sample_index]
            pred = select_prediction_for_image(model(image_tensor.unsqueeze(0).to(device)), 0)
            predictions = decode_predictions_for_image(
                pred=pred,
                image_size=image_size,
                num_classes=num_classes,
                num_boxes=num_boxes,
                anchors=anchors,
                box_parameterization=box_parameterization,
                score_threshold=score_threshold,
                top_k=top_k,
                nms_iou_threshold=nms_iou_threshold,
                score_alpha=score_alpha,
                score_beta=score_beta,
            )

            image_id = int(target["sample_id"].split("_")[-1])
            original_height, original_width = target["original_size"].tolist()
            scale_x = float(original_width) / float(image_size)
            scale_y = float(original_height) / float(image_size)
            for prediction in predictions:
                x1, y1, x2, y2 = prediction["box_xyxy"]
                x1 = max(0.0, min(float(x1) * scale_x, float(original_width)))
                x2 = max(0.0, min(float(x2) * scale_x, float(original_width)))
                y1 = max(0.0, min(float(y1) * scale_y, float(original_height)))
                y2 = max(0.0, min(float(y2) * scale_y, float(original_height)))
                results.append(
                    {
                        "image_id": image_id,
                        "category_id": contiguous_to_original[int(prediction["class_id"])],
                        "bbox": [float(x1), float(y1), float(max(x2 - x1, 0.0)), float(max(y2 - y1, 0.0))],
                        "score": float(prediction["score"]),
                    }
                )

    if was_training:
        model.train()

    if not results:
        return {
            "num_samples": float(num_samples),
            "num_predictions": 0.0,
            "coco_ap": 0.0,
            "coco_ap50": 0.0,
            "coco_ap75": 0.0,
            "coco_ar1": 0.0,
            "coco_ar10": 0.0,
            "coco_ar100": 0.0,
        }

    dt_api = gt_api.loadRes(results)
    evaluator = COCOeval(gt_api, dt_api, "bbox")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    return {
        "num_samples": float(num_samples),
        "num_predictions": float(len(results)),
        "coco_ap": float(evaluator.stats[0]),
        "coco_ap50": float(evaluator.stats[1]),
        "coco_ap75": float(evaluator.stats[2]),
        "coco_ap_small": float(evaluator.stats[3]),
        "coco_ap_medium": float(evaluator.stats[4]),
        "coco_ap_large": float(evaluator.stats[5]),
        "coco_ar1": float(evaluator.stats[6]),
        "coco_ar10": float(evaluator.stats[7]),
        "coco_ar100": float(evaluator.stats[8]),
        "coco_ar_small": float(evaluator.stats[9]),
        "coco_ar_medium": float(evaluator.stats[10]),
        "coco_ar_large": float(evaluator.stats[11]),
    }
