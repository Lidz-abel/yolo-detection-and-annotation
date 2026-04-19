"""Prediction decoding helpers used by visualization and later evaluation scripts."""

from __future__ import annotations

import torch

from utils.box_ops import decode_box_predictions


def decode_predictions_for_image(
    pred: torch.Tensor,
    image_size: int,
    num_classes: int,
    num_boxes: int = 1,
    anchors: list[tuple[float, float]] | None = None,
    score_threshold: float = 0.05,
    top_k: int = 10,
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

    decoded_boxes = decode_box_predictions(pred[..., 0:4], anchors=anchor_tensor)[0]
    obj_scores = torch.sigmoid(pred[..., 4])[0]
    cls_scores = torch.sigmoid(pred[..., 5:])[0]
    best_cls_scores, class_ids = cls_scores.max(dim=-1)
    scores = obj_scores * best_cls_scores

    flat_scores = scores.reshape(-1)
    flat_boxes = decoded_boxes.reshape(-1, 4)
    flat_class_ids = class_ids.reshape(-1)

    keep = flat_scores >= score_threshold
    if keep.sum() == 0:
        return []

    flat_scores = flat_scores[keep]
    flat_boxes = flat_boxes[keep]
    flat_class_ids = flat_class_ids[keep]

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
) -> list[dict]:
    """Keep the old single-box entrypoint as a thin wrapper."""
    return decode_predictions_for_image(
        pred=pred,
        image_size=image_size,
        num_classes=num_classes,
        num_boxes=1,
        anchors=None,
        score_threshold=score_threshold,
        top_k=top_k,
    )
