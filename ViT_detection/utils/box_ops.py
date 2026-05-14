"""Box decoding and geometry utilities shared by loss and visualization code."""

from __future__ import annotations

import torch


def _make_grid_like(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Build broadcastable grid indices for one `[B, G, G, ...]` prediction tensor."""
    grid_size_y = tensor.shape[1]
    grid_size_x = tensor.shape[2]
    device = tensor.device
    dtype = tensor.dtype
    grid_y, grid_x = torch.meshgrid(
        torch.arange(grid_size_y, device=device, dtype=dtype),
        torch.arange(grid_size_x, device=device, dtype=dtype),
        indexing="ij",
    )
    extra_dims = max(tensor.dim() - 4, 0)
    for _ in range(extra_dims):
        grid_x = grid_x.unsqueeze(-1)
        grid_y = grid_y.unsqueeze(-1)
    return grid_x.unsqueeze(0), grid_y.unsqueeze(0)


def decode_box_predictions(
    box_pred: torch.Tensor,
    anchors: torch.Tensor | None = None,
    box_parameterization: str = "legacy",
) -> torch.Tensor:
    """Decode one-box or multi-box predictions into normalized `xyxy` boxes."""
    grid_x, grid_y = _make_grid_like(box_pred)
    grid_size = float(box_pred.shape[1])

    raw_tx = box_pred[..., 0]
    raw_ty = box_pred[..., 1]
    raw_tw = box_pred[..., 2]
    raw_th = box_pred[..., 3]

    if box_parameterization == "yolov5":
        # Allow the center to move beyond the current cell and allow the width
        # and height to grow beyond the raw anchor size, as in YOLOv5-style decoding.
        tx = torch.sigmoid(raw_tx) * 2.0 - 0.5
        ty = torch.sigmoid(raw_ty) * 2.0 - 0.5
        tw = (torch.sigmoid(raw_tw) * 2.0).pow(2)
        th = (torch.sigmoid(raw_th) * 2.0).pow(2)
    else:
        tx = torch.sigmoid(raw_tx)
        ty = torch.sigmoid(raw_ty)
        tw = torch.sigmoid(raw_tw)
        th = torch.sigmoid(raw_th)

    cx = (tx + grid_x) / grid_size
    cy = (ty + grid_y) / grid_size
    if anchors is not None:
        anchor_tensor = torch.as_tensor(anchors, dtype=box_pred.dtype, device=box_pred.device)
        view_shape = [1, 1, 1, anchor_tensor.shape[0]]
        anchor_w = anchor_tensor[:, 0].view(*view_shape)
        anchor_h = anchor_tensor[:, 1].view(*view_shape)
        w = tw * anchor_w
        h = th * anchor_h
    else:
        w = tw
        h = th

    x1 = (cx - w / 2.0).clamp(0.0, 1.0)
    y1 = (cy - h / 2.0).clamp(0.0, 1.0)
    x2 = (cx + w / 2.0).clamp(0.0, 1.0)
    y2 = (cy + h / 2.0).clamp(0.0, 1.0)
    return torch.stack([x1, y1, x2, y2], dim=-1)


def decode_single_box_predictions(box_pred: torch.Tensor) -> torch.Tensor:
    """Decode the legacy single-box prediction path into normalized `xyxy` boxes."""
    return decode_box_predictions(box_pred, anchors=None, box_parameterization="legacy")


def target_boxes_to_xyxy(target_box: torch.Tensor) -> torch.Tensor:
    """Decode target `[cell_x, cell_y, w, h]` tensors into normalized `xyxy` boxes."""
    grid_x, grid_y = _make_grid_like(target_box)
    grid_size = float(target_box.shape[1])

    cx = (target_box[..., 0] + grid_x) / grid_size
    cy = (target_box[..., 1] + grid_y) / grid_size
    w = target_box[..., 2]
    h = target_box[..., 3]

    x1 = (cx - w / 2.0).clamp(0.0, 1.0)
    y1 = (cy - h / 2.0).clamp(0.0, 1.0)
    x2 = (cx + w / 2.0).clamp(0.0, 1.0)
    y2 = (cy + h / 2.0).clamp(0.0, 1.0)
    return torch.stack([x1, y1, x2, y2], dim=-1)


def box_iou_xyxy(boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Compute pairwise IoU for two matching-shape `xyxy` box tensors."""
    inter_x1 = torch.maximum(boxes1[..., 0], boxes2[..., 0])
    inter_y1 = torch.maximum(boxes1[..., 1], boxes2[..., 1])
    inter_x2 = torch.minimum(boxes1[..., 2], boxes2[..., 2])
    inter_y2 = torch.minimum(boxes1[..., 3], boxes2[..., 3])

    inter_w = (inter_x2 - inter_x1).clamp(min=0.0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0.0)
    inter_area = inter_w * inter_h

    area1 = ((boxes1[..., 2] - boxes1[..., 0]).clamp(min=0.0) * (boxes1[..., 3] - boxes1[..., 1]).clamp(min=0.0))
    area2 = ((boxes2[..., 2] - boxes2[..., 0]).clamp(min=0.0) * (boxes2[..., 3] - boxes2[..., 1]).clamp(min=0.0))
    union = area1 + area2 - inter_area
    return inter_area / (union + eps)


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Compute Generalized IoU for two matching-shape `xyxy` box tensors."""
    iou = box_iou_xyxy(boxes1, boxes2, eps=eps)

    enclose_x1 = torch.minimum(boxes1[..., 0], boxes2[..., 0])
    enclose_y1 = torch.minimum(boxes1[..., 1], boxes2[..., 1])
    enclose_x2 = torch.maximum(boxes1[..., 2], boxes2[..., 2])
    enclose_y2 = torch.maximum(boxes1[..., 3], boxes2[..., 3])

    enclose_w = (enclose_x2 - enclose_x1).clamp(min=0.0)
    enclose_h = (enclose_y2 - enclose_y1).clamp(min=0.0)
    enclose_area = enclose_w * enclose_h

    area1 = ((boxes1[..., 2] - boxes1[..., 0]).clamp(min=0.0) * (boxes1[..., 3] - boxes1[..., 1]).clamp(min=0.0))
    area2 = ((boxes2[..., 2] - boxes2[..., 0]).clamp(min=0.0) * (boxes2[..., 3] - boxes2[..., 1]).clamp(min=0.0))

    inter_x1 = torch.maximum(boxes1[..., 0], boxes2[..., 0])
    inter_y1 = torch.maximum(boxes1[..., 1], boxes2[..., 1])
    inter_x2 = torch.minimum(boxes1[..., 2], boxes2[..., 2])
    inter_y2 = torch.minimum(boxes1[..., 3], boxes2[..., 3])
    inter_w = (inter_x2 - inter_x1).clamp(min=0.0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0.0)
    inter_area = inter_w * inter_h
    union = area1 + area2 - inter_area

    return iou - (enclose_area - union) / (enclose_area + eps)
