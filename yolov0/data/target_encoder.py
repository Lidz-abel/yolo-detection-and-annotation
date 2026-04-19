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


def _anchor_shape_ratio(box_wh: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    """Measure width-height ratio mismatch against anchors; smaller is better."""
    width_ratio = torch.maximum(
        box_wh[0] / anchors[:, 0].clamp(min=1e-7),
        anchors[:, 0] / box_wh[0].clamp(min=1e-7),
    )
    height_ratio = torch.maximum(
        box_wh[1] / anchors[:, 1].clamp(min=1e-7),
        anchors[:, 1] / box_wh[1].clamp(min=1e-7),
    )
    return torch.maximum(width_ratio, height_ratio)


def encode_target(
    boxes,
    labels,
    image_size=320,
    grid_size=10,
    num_classes=80,
    num_boxes=1,
    anchors=None,
    anchor_positive_iou=0.25,
    anchor_ignore_iou=0.5,
    anchor_match_metric="iou",
    anchor_shape_ratio=4.0,
    anchor_ignore_shape_ratio=None,
):
    """Map resized xyxy boxes to a grid with box, class, and objectness targets."""
    if num_boxes > 1 and anchors:
        target_cls = torch.zeros(grid_size, grid_size, num_boxes, num_classes)
        target_box = torch.zeros(grid_size, grid_size, num_boxes, 4)
        object_mask = torch.zeros(grid_size, grid_size, num_boxes)
        target_obj = torch.zeros(grid_size, grid_size, num_boxes)
        noobj_mask = torch.ones(grid_size, grid_size, num_boxes)
        anchors_tensor = torch.tensor(anchors, dtype=torch.float32)
    else:
        target_cls = torch.zeros(grid_size, grid_size, num_classes)
        target_box = torch.zeros(grid_size, grid_size, 4)
        object_mask = torch.zeros(grid_size, grid_size)
        target_obj = torch.zeros(grid_size, grid_size)
        noobj_mask = torch.ones(grid_size, grid_size)
        box_area_map = torch.zeros(grid_size, grid_size)
    collision_count = torch.tensor(0.0)
    ignored_count = torch.tensor(0.0)
    dropped_gt_count = torch.tensor(0.0)

    if num_boxes > 1 and anchors:
        cell_assignments: dict[tuple[int, int], list[dict[str, torch.Tensor | int | float]]] = {}
    else:
        cell_assignments = {}

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
            cell_key = (grid_y, grid_x)
            cell_assignments.setdefault(cell_key, []).append(
                {
                    "label": int(label),
                    "box_target": torch.tensor([cell_x, cell_y, w, h], dtype=torch.float32),
                    "box_wh": torch.tensor([w, h], dtype=torch.float32),
                }
            )
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

    if num_boxes > 1 and anchors:
        positive_iou = float(anchor_positive_iou)
        ignore_iou = float(anchor_ignore_iou)
        match_metric = str(anchor_match_metric).lower()
        positive_ratio = float(anchor_shape_ratio)
        ignore_ratio = (
            float(anchor_ignore_shape_ratio)
            if anchor_ignore_shape_ratio is not None
            else float(anchor_shape_ratio) + 1.5
        )

        for (grid_y, grid_x), entries in cell_assignments.items():
            if len(entries) > 1:
                collision_count += float(len(entries) - 1)

            gt_count = len(entries)
            if match_metric == "shape_ratio":
                fit_matrix = torch.stack(
                    [_anchor_shape_ratio(entry["box_wh"], anchors_tensor) for entry in entries],
                    dim=0,
                )
                fit_is_better = "lower"
            else:
                fit_matrix = torch.stack(
                    [_anchor_iou(entry["box_wh"], anchors_tensor) for entry in entries],
                    dim=0,
                )
                fit_is_better = "higher"
            assigned_anchor_for_gt = [-1] * gt_count
            assigned_gt_for_anchor = [-1] * num_boxes

            # Pass 1: give each GT one anchor when possible, preferring the best IoU fit.
            if fit_is_better == "lower":
                gt_priority = sorted(
                    range(gt_count),
                    key=lambda gt_idx: float(fit_matrix[gt_idx].min().item()),
                )
            else:
                gt_priority = sorted(
                    range(gt_count),
                    key=lambda gt_idx: float(fit_matrix[gt_idx].max().item()),
                    reverse=True,
                )
            for gt_idx in gt_priority:
                anchor_order = torch.argsort(
                    fit_matrix[gt_idx],
                    descending=(fit_is_better == "higher"),
                ).tolist()
                for anchor_idx in anchor_order:
                    if assigned_gt_for_anchor[anchor_idx] == -1:
                        assigned_anchor_for_gt[gt_idx] = anchor_idx
                        assigned_gt_for_anchor[anchor_idx] = gt_idx
                        break
                if assigned_anchor_for_gt[gt_idx] == -1:
                    dropped_gt_count += 1

            # Pass 2: expand positives to additional good-fitting anchors that remain free.
            extra_candidates = []
            for gt_idx in range(gt_count):
                for anchor_idx in range(num_boxes):
                    if assigned_gt_for_anchor[anchor_idx] != -1:
                        continue
                    score = float(fit_matrix[gt_idx, anchor_idx].item())
                    if fit_is_better == "lower":
                        if score <= positive_ratio:
                            extra_candidates.append((score, gt_idx, anchor_idx))
                    else:
                        if score >= positive_iou:
                            extra_candidates.append((score, gt_idx, anchor_idx))
            extra_candidates.sort(reverse=(fit_is_better == "higher"))
            for _, gt_idx, anchor_idx in extra_candidates:
                if assigned_gt_for_anchor[anchor_idx] == -1:
                    assigned_gt_for_anchor[anchor_idx] = gt_idx

            if fit_is_better == "lower":
                best_fit_per_anchor = fit_matrix.min(dim=0).values
            else:
                best_fit_per_anchor = fit_matrix.max(dim=0).values
            for anchor_idx in range(num_boxes):
                gt_idx = assigned_gt_for_anchor[anchor_idx]
                if gt_idx != -1:
                    entry = entries[gt_idx]
                    object_mask[grid_y, grid_x, anchor_idx] = 1
                    target_obj[grid_y, grid_x, anchor_idx] = 1
                    noobj_mask[grid_y, grid_x, anchor_idx] = 0
                    target_cls[grid_y, grid_x, anchor_idx].zero_()
                    target_cls[grid_y, grid_x, anchor_idx, entry["label"]] = 1
                    target_box[grid_y, grid_x, anchor_idx] = entry["box_target"]
                else:
                    best_fit = float(best_fit_per_anchor[anchor_idx].item())
                    if fit_is_better == "lower":
                        if best_fit <= ignore_ratio:
                            noobj_mask[grid_y, grid_x, anchor_idx] = 0
                            ignored_count += 1
                    elif best_fit >= ignore_iou:
                        noobj_mask[grid_y, grid_x, anchor_idx] = 0
                        ignored_count += 1

    return (
        target_cls,
        target_box,
        target_obj,
        object_mask,
        noobj_mask,
        collision_count,
        ignored_count,
        dropped_gt_count,
    )
