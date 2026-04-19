"""Encode detection annotations into YOLO-style grid supervision tensors."""

import torch


def _anchor_iou(box_wh: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    """Measure width-height IoU against the configured anchors."""
    inter_w = torch.minimum(box_wh[0], anchors[:, 0])
    inter_h = torch.minimum(box_wh[1], anchors[:, 1])
    intersection = inter_w * inter_h
    box_area = box_wh[0] * box_wh[1]
    anchor_area = anchors[:, 0] * anchors[:, 1]
    union = box_area + anchor_area - intersection
    return intersection / union.clamp(min=1e-7)


def encode_target(
    boxes,
    labels,
    image_size=320,
    grid_size=10,
    num_classes=80,
    num_boxes=1,
    anchors=None,
    anchor_ignore_iou=0.5,
):
    """Map resized xyxy boxes to a grid with box, class, and objectness targets."""
    if num_boxes > 1 and anchors:
        target_cls = torch.zeros(grid_size, grid_size, num_boxes, num_classes)
        target_box = torch.zeros(grid_size, grid_size, num_boxes, 4)
        object_mask = torch.zeros(grid_size, grid_size, num_boxes)
        target_obj = torch.zeros(grid_size, grid_size, num_boxes)
        noobj_mask = torch.ones(grid_size, grid_size, num_boxes)
        assigned_iou_map = torch.full((grid_size, grid_size, num_boxes), -1.0)
        anchors_tensor = torch.tensor(anchors, dtype=torch.float32)
    else:
        target_cls = torch.zeros(grid_size, grid_size, num_classes)
        target_box = torch.zeros(grid_size, grid_size, 4)
        object_mask = torch.zeros(grid_size, grid_size)
        target_obj = torch.zeros(grid_size, grid_size)
        noobj_mask = torch.ones(grid_size, grid_size)
        box_area_map = torch.zeros(grid_size, grid_size)
    collision_count = torch.tensor(0.0)

    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = box

        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = x2 - x1
        h = y2 - y1

        cx = cx / image_size
        cy = cy / image_size
        w = w / image_size
        h = h / image_size

        grid_x = int(cx * grid_size)
        grid_y = int(cy * grid_size)
        grid_x = min(grid_x, grid_size - 1)
        grid_y = min(grid_y, grid_size - 1)

        cell_x = cx * grid_size - grid_x
        cell_y = cy * grid_size - grid_y
        box_area = w * h

        if num_boxes > 1 and anchors:
            box_wh = torch.tensor([w, h], dtype=torch.float32)
            anchor_ious = _anchor_iou(box_wh, anchors_tensor)
            best_anchor = int(torch.argmax(anchor_ious).item())
            ignore_mask = anchor_ious >= float(anchor_ignore_iou)
            noobj_mask[grid_y, grid_x, ignore_mask] = 0

            if object_mask[grid_y, grid_x, best_anchor] > 0:
                collision_count += 1

            # Keep the object-anchor pairing with the stronger width-height fit.
            if assigned_iou_map[grid_y, grid_x, best_anchor] <= anchor_ious[best_anchor]:
                object_mask[grid_y, grid_x, best_anchor] = 1
                target_obj[grid_y, grid_x, best_anchor] = 1
                noobj_mask[grid_y, grid_x, best_anchor] = 0
                target_cls[grid_y, grid_x, best_anchor].zero_()
                target_cls[grid_y, grid_x, best_anchor, int(label)] = 1
                target_box[grid_y, grid_x, best_anchor] = torch.tensor([cell_x, cell_y, w, h])
                assigned_iou_map[grid_y, grid_x, best_anchor] = anchor_ious[best_anchor]
        else:
            if object_mask[grid_y, grid_x] > 0:
                collision_count += 1

            # Keep the larger object when multiple boxes fall into the same cell.
            if object_mask[grid_y, grid_x] == 0 or box_area >= box_area_map[grid_y, grid_x]:
                object_mask[grid_y, grid_x] = 1
                target_obj[grid_y, grid_x] = 1
                noobj_mask[grid_y, grid_x] = 0
                target_cls[grid_y, grid_x].zero_()
                target_cls[grid_y, grid_x, int(label)] = 1
                target_box[grid_y, grid_x] = torch.tensor([cell_x, cell_y, w, h])
                box_area_map[grid_y, grid_x] = box_area

    return target_cls, target_box, target_obj, object_mask, noobj_mask, collision_count
