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


def select_prediction_for_image(pred, image_index: int = 0):
    """Extract one image prediction from a batch tensor or multi-scale dict."""
    if isinstance(pred, dict):
        return {name: value[image_index].detach().cpu() for name, value in pred.items()}
    return pred[image_index].detach().cpu()


def _flatten_prediction_outputs(
    pred: torch.Tensor,
    num_classes: int,
    num_boxes: int,
    anchors: list[tuple[float, float]] | None,
    box_parameterization: str,
    score_alpha: float,
    score_beta: float,
):
    """Decode one scale and return flat boxes, scores, and class ids."""
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
    scores = obj_scores.pow(score_alpha) * best_cls_scores.pow(score_beta)
    return decoded_boxes.reshape(-1, 4), scores.reshape(-1), class_ids.reshape(-1)


def _anchors_for_scale(
    anchors: list[tuple[float, float]] | dict[str, list[tuple[float, float]]] | None,
    scale_name: str | None,
):
    """Pick one scale-specific anchor list when a multiscale anchor map is provided."""
    if isinstance(anchors, dict):
        if scale_name is None:
            return []
        return anchors.get(scale_name, [])
    return anchors


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
    if isinstance(pred, dict):
        flat_boxes_parts = []
        flat_scores_parts = []
        flat_class_parts = []
        for scale_name, scale_pred in pred.items():
            if scale_pred.dim() != 3:
                raise ValueError(f"Expected one image prediction tensor, got shape {tuple(scale_pred.shape)}")
            scale_boxes, scale_scores, scale_classes = _flatten_prediction_outputs(
                pred=scale_pred,
                num_classes=num_classes,
                num_boxes=num_boxes,
                anchors=_anchors_for_scale(anchors, scale_name),
                box_parameterization=box_parameterization,
                score_alpha=score_alpha,
                score_beta=score_beta,
            )
            flat_boxes_parts.append(scale_boxes)
            flat_scores_parts.append(scale_scores)
            flat_class_parts.append(scale_classes)
        flat_boxes = torch.cat(flat_boxes_parts, dim=0)
        flat_scores = torch.cat(flat_scores_parts, dim=0)
        flat_class_ids = torch.cat(flat_class_parts, dim=0)
    else:
        if pred.dim() != 3:
            raise ValueError(f"Expected one image prediction tensor, got shape {tuple(pred.shape)}")
        flat_boxes, flat_scores, flat_class_ids = _flatten_prediction_outputs(
            pred=pred,
            num_classes=num_classes,
            num_boxes=num_boxes,
            anchors=anchors,
            box_parameterization=box_parameterization,
            score_alpha=score_alpha,
            score_beta=score_beta,
        )

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
