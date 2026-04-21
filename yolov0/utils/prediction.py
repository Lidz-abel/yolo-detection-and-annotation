"""Prediction decoding helpers used by visualization and evaluation scripts."""

from __future__ import annotations

import torch

from utils.box_ops import box_iou_xyxy, decode_box_predictions


def _nms_indices(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    """Run a simple score-sorted NMS over one set of boxes without extra deps."""
    if boxes.numel() == 0:
        return boxes.new_zeros((0,), dtype=torch.long)

    order = torch.argsort(scores, descending=True)
    keep: list[int] = []
    while order.numel() > 0:
        current = int(order[0].item())
        keep.append(current)
        if order.numel() == 1:
            break
        current_box = boxes[current].unsqueeze(0)
        remaining = order[1:]
        ious = box_iou_xyxy(current_box.expand(remaining.numel(), -1), boxes[remaining])
        order = remaining[ious <= iou_threshold]
    return boxes.new_tensor(keep, dtype=torch.long)


def decode_predictions_for_image(
    pred: torch.Tensor,
    image_size: int,
    num_classes: int,
    num_boxes: int = 1,
    anchors: list[tuple[float, float]] | None = None,
    box_parameterization: str = "legacy",
    score_threshold: float = 0.05,
    top_k: int = 10,
    nms_iou_threshold: float = 0.5,
    score_alpha: float = 1.0,
    score_beta: float = 1.0,
) -> list[dict]:
    """Decode one image prediction tensor into ranked box dictionaries."""
    if pred.dim() != 3:
        raise ValueError(f"Expected one image prediction tensor, got shape {tuple(pred.shape)}")

    pred = pred.unsqueeze(0)
    if num_boxes > 1:
        pred = pred.view(*pred.shape[:3], num_boxes, num_classes + 5)
        anchor_tensor = pred.new_tensor(anchors) if anchors else None
    else:
        anchor_tensor = None

    decoded_boxes = decode_box_predictions(
        pred[..., 0:4],
        anchors=anchor_tensor,
        box_parameterization=box_parameterization,
    )[0]
    obj_scores = torch.sigmoid(pred[..., 4])[0]
    cls_scores = torch.sigmoid(pred[..., 5:])[0]
    best_cls_scores, class_ids = cls_scores.max(dim=-1)
    # Round 6A uses a power-product score so objectness can dominate ranking
    # without retraining the detector. The default exponents keep legacy
    # behavior unchanged.
    scores = obj_scores.pow(score_alpha) * best_cls_scores.pow(score_beta)

    flat_scores = scores.reshape(-1)
    flat_boxes = decoded_boxes.reshape(-1, 4)
    flat_class_ids = class_ids.reshape(-1)

    keep = flat_scores >= score_threshold
    if keep.sum() == 0:
        return []

    flat_scores = flat_scores[keep]
    flat_boxes = flat_boxes[keep]
    flat_class_ids = flat_class_ids[keep]

    if nms_iou_threshold > 0:
        kept_orders = []
        for class_id in torch.unique(flat_class_ids).tolist():
            class_mask = flat_class_ids == class_id
            class_indices = torch.nonzero(class_mask, as_tuple=False).squeeze(1)
            class_keep = _nms_indices(
                flat_boxes[class_mask],
                flat_scores[class_mask],
                iou_threshold=nms_iou_threshold,
            )
            kept_orders.append(class_indices[class_keep])
        order = torch.cat(kept_orders, dim=0) if kept_orders else flat_scores.new_zeros((0,), dtype=torch.long)
        if order.numel() > 0:
            order = order[torch.argsort(flat_scores[order], descending=True)]
    else:
        order = torch.argsort(flat_scores, descending=True)
    order = order[:top_k]

    predictions = []
    for index in order.tolist():
        x1, y1, x2, y2 = flat_boxes[index].tolist()
        predictions.append(
            {
                "box_xyxy": [
                    x1 * image_size,
                    y1 * image_size,
                    x2 * image_size,
                    y2 * image_size,
                ],
                "score": float(flat_scores[index].item()),
                "class_id": int(flat_class_ids[index].item()),
            }
        )
    return predictions


def decode_single_box_predictions_for_image(
    pred: torch.Tensor,
    image_size: int,
    num_classes: int,
    score_threshold: float = 0.05,
    top_k: int = 10,
    nms_iou_threshold: float = 0.5,
    score_alpha: float = 1.0,
    score_beta: float = 1.0,
) -> list[dict]:
    """Keep the old single-box entrypoint as a thin wrapper."""
    return decode_predictions_for_image(
        pred=pred,
        image_size=image_size,
        num_classes=num_classes,
        num_boxes=1,
        anchors=None,
        box_parameterization="legacy",
        score_threshold=score_threshold,
        top_k=top_k,
        nms_iou_threshold=nms_iou_threshold,
        score_alpha=score_alpha,
        score_beta=score_beta,
    )
