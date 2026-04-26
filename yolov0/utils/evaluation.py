"""Detection metrics used for formal yolov0 checkpoint-6 comparisons."""

from __future__ import annotations

from collections import defaultdict

import torch

from utils.box_ops import box_iou_xyxy
from utils.prediction import decode_predictions_for_image, select_prediction_for_image


def _compute_ap(recall: torch.Tensor, precision: torch.Tensor) -> float:
    """Compute area-under-curve AP from one precision-recall trace."""
    mrec = torch.cat([torch.tensor([0.0]), recall, torch.tensor([1.0])])
    mpre = torch.cat([torch.tensor([0.0]), precision, torch.tensor([0.0])])
    for index in range(mpre.numel() - 1, 0, -1):
        mpre[index - 1] = torch.maximum(mpre[index - 1], mpre[index])
    change = torch.nonzero(mrec[1:] != mrec[:-1], as_tuple=False).squeeze(1)
    return float(torch.sum((mrec[change + 1] - mrec[change]) * mpre[change + 1]).item())


def evaluate_detector(
    model,
    dataset,
    device,
    num_classes: int,
    num_boxes: int = 1,
    anchors: list[tuple[float, float]] | None = None,
    box_parameterization: str = "legacy",
    max_samples: int = 0,
    score_threshold: float = 0.05,
    top_k: int = 100,
    nms_iou_threshold: float = 0.5,
    map_iou_threshold: float = 0.5,
    score_alpha: float = 1.0,
    score_beta: float = 1.0,
) -> dict[str, float]:
    """Evaluate mAP@0.5, precision, and recall on one dataset split."""
    was_training = model.training
    model.eval()

    num_samples = len(dataset) if max_samples <= 0 else min(int(max_samples), len(dataset))
    gt_count_by_class = defaultdict(int)
    matched_gt = {}
    predictions_by_class = defaultdict(list)
    total_predictions = 0

    with torch.no_grad():
        for sample_index in range(num_samples):
            image_tensor, target = dataset[sample_index]
            pred = select_prediction_for_image(model(image_tensor.unsqueeze(0).to(device)), 0)
            predictions = decode_predictions_for_image(
                pred=pred,
                image_size=int(target["resized_size"][0].item()),
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
            total_predictions += len(predictions)

            gt_boxes = target["boxes"].float()
            gt_labels = target["labels"].long()
            for gt_index, label in enumerate(gt_labels.tolist()):
                gt_count_by_class[label] += 1
                matched_gt[(sample_index, gt_index)] = False

            for prediction in predictions:
                class_id = int(prediction["class_id"])
                prediction_box = torch.tensor(prediction["box_xyxy"], dtype=torch.float32)

                candidate_indices = torch.nonzero(gt_labels == class_id, as_tuple=False).squeeze(1)
                best_iou = 0.0
                best_gt_index = None
                if candidate_indices.numel() > 0:
                    candidate_boxes = gt_boxes[candidate_indices]
                    repeated_pred = prediction_box.unsqueeze(0).expand(candidate_boxes.shape[0], -1)
                    ious = box_iou_xyxy(repeated_pred, candidate_boxes)
                    best_pos = int(torch.argmax(ious).item())
                    best_iou = float(ious[best_pos].item())
                    best_gt_index = int(candidate_indices[best_pos].item())

                predictions_by_class[class_id].append(
                    {
                        "sample_index": sample_index,
                        "score": float(prediction["score"]),
                        "best_iou": best_iou,
                        "best_gt_index": best_gt_index,
                    }
                )

    if was_training:
        model.train()

    ap_values = []
    global_tp = 0
    global_fp = 0
    total_gt = sum(gt_count_by_class.values())

    for class_id, class_predictions in predictions_by_class.items():
        class_predictions = sorted(class_predictions, key=lambda item: item["score"], reverse=True)
        if gt_count_by_class[class_id] == 0:
            continue

        tp = torch.zeros(len(class_predictions), dtype=torch.float32)
        fp = torch.zeros(len(class_predictions), dtype=torch.float32)

        for pred_index, prediction in enumerate(class_predictions):
            gt_index = prediction["best_gt_index"]
            if (
                gt_index is not None
                and prediction["best_iou"] >= map_iou_threshold
                and not matched_gt[(prediction["sample_index"], gt_index)]
            ):
                matched_gt[(prediction["sample_index"], gt_index)] = True
                tp[pred_index] = 1.0
                global_tp += 1
            else:
                fp[pred_index] = 1.0
                global_fp += 1

        tp_cum = torch.cumsum(tp, dim=0)
        fp_cum = torch.cumsum(fp, dim=0)
        precision = tp_cum / (tp_cum + fp_cum).clamp(min=1e-7)
        recall = tp_cum / max(float(gt_count_by_class[class_id]), 1.0)
        ap_values.append(_compute_ap(recall, precision))

    precision_value = global_tp / max(global_tp + global_fp, 1)
    recall_value = global_tp / max(total_gt, 1)
    mean_ap = sum(ap_values) / max(len(ap_values), 1)
    return {
        "num_samples": float(num_samples),
        "num_predictions": float(total_predictions),
        "num_ground_truth": float(total_gt),
        "true_positives": float(global_tp),
        "false_positives": float(global_fp),
        "precision": float(precision_value),
        "recall": float(recall_value),
        "map50": float(mean_ap),
        "num_evaluated_classes": float(len(ap_values)),
    }
