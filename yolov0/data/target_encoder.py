"""Encode detection annotations into YOLO-style grid supervision tensors."""

import torch


def encode_target(
    boxes,
    labels,
    image_size=320,
    grid_size=10,
    num_classes=80,
):
    """Map resized xyxy boxes to a grid, one-hot classes, and object mask."""
    target_cls = torch.zeros(grid_size, grid_size, num_classes)
    target_box = torch.zeros(grid_size, grid_size, 4)
    object_mask = torch.zeros(grid_size, grid_size)

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

        object_mask[grid_y, grid_x] = 1
        target_cls[grid_y, grid_x, int(label)] = 1
        target_box[grid_y, grid_x] = torch.tensor([cell_x, cell_y, w, h])

    return target_cls, target_box, object_mask
