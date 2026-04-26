"""YOLO-style loss with objectness, class BCE, and GIoU box loss."""

from __future__ import annotations

import torch
import torch.nn as nn

from utils.box_ops import box_iou_xyxy, decode_box_predictions, generalized_box_iou, target_boxes_to_xyxy


class YOLOLoss(nn.Module):
    """Compute the practical single-box or multi-box detection loss for yolov0."""

    def __init__(
        self,
        num_classes=80,
        lambda_box=5.0,
        lambda_obj=1.0,
        lambda_noobj=0.5,
        lambda_cls=1.0,
        num_boxes=1,
        anchors=None,
        anchors_by_level=None,
        box_parameterization="legacy",
        soft_objectness_target="hard",
        soft_objectness_min=0.0,
        soft_classification_target="hard",
        cls_loss_mode="bce",
        varifocal_alpha=0.75,
        varifocal_gamma=2.0,
        assignment_strategy="static",
        dynamic_topk=2,
        dynamic_center_radius=1,
        dynamic_box_cost=3.0,
        dynamic_cls_cost=1.0,
        dynamic_ignore_iou=0.5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_box = lambda_box
        self.lambda_obj = lambda_obj
        self.lambda_noobj = lambda_noobj
        self.lambda_cls = lambda_cls
        self.num_boxes = num_boxes
        self.anchors = anchors or []
        self.anchors_by_level = anchors_by_level or {}
        self.box_parameterization = box_parameterization
        self.soft_objectness_target = soft_objectness_target
        self.soft_objectness_min = soft_objectness_min
        self.soft_classification_target = soft_classification_target
        self.cls_loss_mode = cls_loss_mode
        self.varifocal_alpha = varifocal_alpha
        self.varifocal_gamma = varifocal_gamma
        self.assignment_strategy = assignment_strategy
        self.dynamic_topk = dynamic_topk
        self.dynamic_center_radius = dynamic_center_radius
        self.dynamic_box_cost = dynamic_box_cost
        self.dynamic_cls_cost = dynamic_cls_cost
        self.dynamic_ignore_iou = dynamic_ignore_iou
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def _build_dynamic_targets(self, pred_cls, decoded_pred_boxes, targets):
        """Assign positives dynamically from current predictions and GT boxes."""
        batch_size, grid_h, grid_w, num_boxes, _ = decoded_pred_boxes.shape
        device = decoded_pred_boxes.device
        dtype = decoded_pred_boxes.dtype
        num_classes = pred_cls.shape[-1]

        target_box = torch.zeros(batch_size, grid_h, grid_w, num_boxes, 4, device=device, dtype=dtype)
        target_obj = torch.zeros(batch_size, grid_h, grid_w, num_boxes, device=device, dtype=dtype)
        target_cls = torch.zeros(batch_size, grid_h, grid_w, num_boxes, num_classes, device=device, dtype=dtype)
        object_mask = torch.zeros(batch_size, grid_h, grid_w, num_boxes, device=device, dtype=dtype)
        noobj_mask = torch.ones(batch_size, grid_h, grid_w, num_boxes, device=device, dtype=dtype)
        collision_count = torch.zeros(batch_size, device=device, dtype=dtype)
        ignored_count = torch.zeros(batch_size, device=device, dtype=dtype)
        dropped_gt_count = torch.zeros(batch_size, device=device, dtype=dtype)

        slot_grid_y, slot_grid_x, slot_anchor = self._make_slot_indices(
            grid_h=grid_h,
            grid_w=grid_w,
            num_boxes=num_boxes,
            device=device,
            dtype=dtype,
        )
        num_slots = slot_grid_y.numel()

        with torch.no_grad():
            pred_boxes_detached = decoded_pred_boxes.detach().view(batch_size, num_slots, 4)
            pred_cls_detached = pred_cls.detach().view(batch_size, num_slots, num_classes)
            resized_sizes = targets["resized_size"].to(device)

            for batch_index in range(batch_size):
                gt_boxes_abs = targets["boxes"][batch_index].to(device=device, dtype=dtype)
                gt_labels = targets["labels"][batch_index].to(device=device)
                if gt_boxes_abs.numel() == 0:
                    continue

                image_h = float(resized_sizes[batch_index, 0].item())
                image_w = float(resized_sizes[batch_index, 1].item())
                gt_boxes = gt_boxes_abs.clone()
                gt_boxes[:, [0, 2]] /= max(image_w, 1.0)
                gt_boxes[:, [1, 3]] /= max(image_h, 1.0)

                gt_cx = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2.0
                gt_cy = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2.0
                gt_w = gt_boxes[:, 2] - gt_boxes[:, 0]
                gt_h = gt_boxes[:, 3] - gt_boxes[:, 1]
                gt_grid_x = torch.clamp((gt_cx * grid_w).long(), 0, grid_w - 1)
                gt_grid_y = torch.clamp((gt_cy * grid_h).long(), 0, grid_h - 1)

                unique_cells, counts = torch.unique(
                    torch.stack([gt_grid_y, gt_grid_x], dim=1),
                    dim=0,
                    return_counts=True,
                )
                if unique_cells.numel() > 0:
                    collision_count[batch_index] = torch.clamp(counts.float() - 1.0, min=0.0).sum()

                gt_candidate_lists: list[list[tuple[float, int]]] = []
                pred_boxes_image = pred_boxes_detached[batch_index]
                pred_cls_image = pred_cls_detached[batch_index]

                for gt_index in range(gt_boxes.shape[0]):
                    center_mask = (
                        (slot_grid_x - gt_grid_x[gt_index]).abs() <= self.dynamic_center_radius
                    ) & (
                        (slot_grid_y - gt_grid_y[gt_index]).abs() <= self.dynamic_center_radius
                    )
                    candidate_indices = torch.nonzero(center_mask, as_tuple=False).flatten()
                    if candidate_indices.numel() == 0:
                        same_cell_mask = (slot_grid_x == gt_grid_x[gt_index]) & (slot_grid_y == gt_grid_y[gt_index])
                        candidate_indices = torch.nonzero(same_cell_mask, as_tuple=False).flatten()
                    if candidate_indices.numel() == 0:
                        candidate_indices = torch.arange(num_slots, device=device)

                    gt_box = gt_boxes[gt_index].unsqueeze(0).expand(candidate_indices.numel(), 4)
                    iou_scores = box_iou_xyxy(pred_boxes_image[candidate_indices], gt_box)
                    gt_onehot = torch.zeros(num_classes, device=device, dtype=dtype)
                    gt_onehot[int(gt_labels[gt_index].item())] = 1.0
                    cls_cost = self.bce(
                        pred_cls_image[candidate_indices],
                        gt_onehot.unsqueeze(0).expand(candidate_indices.numel(), num_classes),
                    ).mean(dim=-1)
                    total_cost = self.dynamic_box_cost * (1.0 - iou_scores) + self.dynamic_cls_cost * cls_cost
                    sorted_cost, order = torch.sort(total_cost, descending=False)
                    ordered_candidates = candidate_indices[order]
                    gt_candidate_lists.append(
                        [
                            (float(sorted_cost[idx].item()), int(ordered_candidates[idx].item()))
                            for idx in range(min(self.dynamic_topk, ordered_candidates.numel()))
                        ]
                    )

                assigned_slots: set[int] = set()
                assigned_pairs: list[tuple[int, int]] = []
                best_candidate_costs = [
                    candidates[0][0] if candidates else float("inf")
                    for candidates in gt_candidate_lists
                ]
                gt_order = sorted(range(len(gt_candidate_lists)), key=lambda idx: best_candidate_costs[idx])

                # Pass 1 keeps the strongest candidate for every GT whenever possible.
                for gt_index in gt_order:
                    picked_slot = None
                    for _, slot_index in gt_candidate_lists[gt_index]:
                        if slot_index not in assigned_slots:
                            picked_slot = slot_index
                            break
                    if picked_slot is None:
                        dropped_gt_count[batch_index] += 1.0
                        continue
                    assigned_slots.add(picked_slot)
                    assigned_pairs.append((gt_index, picked_slot))

                # Pass 2 expands to a small number of extra low-cost slots per GT.
                for gt_index in gt_order:
                    current_slots = [slot for pair_gt, slot in assigned_pairs if pair_gt == gt_index]
                    allowed_slots = max(self.dynamic_topk - len(current_slots), 0)
                    if allowed_slots == 0:
                        continue
                    for _, slot_index in gt_candidate_lists[gt_index]:
                        if slot_index in assigned_slots:
                            continue
                        assigned_slots.add(slot_index)
                        assigned_pairs.append((gt_index, slot_index))
                        allowed_slots -= 1
                        if allowed_slots == 0:
                            break

                for gt_index, slot_index in assigned_pairs:
                    grid_y = int(slot_grid_y[slot_index].item())
                    grid_x = int(slot_grid_x[slot_index].item())
                    anchor_index = int(slot_anchor[slot_index].item())
                    cell_x = gt_cx[gt_index] * grid_w - float(grid_x)
                    cell_y = gt_cy[gt_index] * grid_h - float(grid_y)
                    target_box[batch_index, grid_y, grid_x, anchor_index] = torch.tensor(
                        [cell_x, cell_y, gt_w[gt_index], gt_h[gt_index]],
                        dtype=dtype,
                        device=device,
                    )
                    target_obj[batch_index, grid_y, grid_x, anchor_index] = 1.0
                    object_mask[batch_index, grid_y, grid_x, anchor_index] = 1.0
                    noobj_mask[batch_index, grid_y, grid_x, anchor_index] = 0.0
                    target_cls[batch_index, grid_y, grid_x, anchor_index, int(gt_labels[gt_index].item())] = 1.0

                if gt_boxes.shape[0] > 0:
                    gt_boxes_expanded = gt_boxes.unsqueeze(0).expand(num_slots, gt_boxes.shape[0], 4)
                    pred_boxes_expanded = pred_boxes_image.unsqueeze(1).expand(num_slots, gt_boxes.shape[0], 4)
                    slot_gt_iou = box_iou_xyxy(pred_boxes_expanded, gt_boxes_expanded)
                    max_iou_per_slot = slot_gt_iou.max(dim=1).values
                    ignore_mask = (max_iou_per_slot >= self.dynamic_ignore_iou)
                    for slot_index in torch.nonzero(ignore_mask, as_tuple=False).flatten():
                        if int(slot_index.item()) in assigned_slots:
                            continue
                        grid_y = int(slot_grid_y[slot_index].item())
                        grid_x = int(slot_grid_x[slot_index].item())
                        anchor_index = int(slot_anchor[slot_index].item())
                        noobj_mask[batch_index, grid_y, grid_x, anchor_index] = 0.0
                        ignored_count[batch_index] += 1.0

        return {
            "target_box": target_box,
            "target_obj": target_obj,
            "target_cls": target_cls,
            "object_mask": object_mask,
            "noobj_mask": noobj_mask,
            "collision_count": collision_count,
            "ignored_count": ignored_count,
            "dropped_gt_count": dropped_gt_count,
        }

    @staticmethod
    def _make_slot_indices(grid_h, grid_w, num_boxes, device, dtype):
        """Create flattened `(grid_y, grid_x, anchor_idx)` lookup tensors."""
        grid_y, grid_x = torch.meshgrid(
            torch.arange(grid_h, device=device, dtype=dtype),
            torch.arange(grid_w, device=device, dtype=dtype),
            indexing="ij",
        )
        grid_y = grid_y.unsqueeze(-1).expand(grid_h, grid_w, num_boxes).reshape(-1)
        grid_x = grid_x.unsqueeze(-1).expand(grid_h, grid_w, num_boxes).reshape(-1)
        anchor_idx = torch.arange(num_boxes, device=device, dtype=dtype).view(1, 1, num_boxes)
        anchor_idx = anchor_idx.expand(grid_h, grid_w, num_boxes).reshape(-1)
        return grid_y.long(), grid_x.long(), anchor_idx.long()

    def _anchors_for_scale(self, scale_name: str | None):
        """Return the correct anchor list for one scale or the global fallback."""
        if scale_name is not None and scale_name in self.anchors_by_level:
            return self.anchors_by_level[scale_name]
        return self.anchors

    def _forward_single_scale(self, pred, targets, scale_name: str | None = None):
        """Return a dict of loss tensors and monitoring metrics for one scale."""
        if self.num_boxes > 1:
            pred = pred.view(*pred.shape[:3], self.num_boxes, self.num_classes + 5)
        pred_box = pred[..., 0:4]
        pred_obj = pred[..., 4]
        pred_cls = pred[..., 5:]

        anchor_tensor = None
        scale_anchors = self._anchors_for_scale(scale_name)
        if self.num_boxes > 1 and scale_anchors:
            anchor_tensor = torch.tensor(scale_anchors, dtype=pred_box.dtype, device=pred_box.device)
        decoded_pred_boxes = decode_box_predictions(
            pred_box,
            anchors=anchor_tensor,
            box_parameterization=self.box_parameterization,
        )

        if self.assignment_strategy == "dynamic_cost" and self.num_boxes > 1:
            dynamic_targets = self._build_dynamic_targets(
                pred_cls=pred_cls,
                decoded_pred_boxes=decoded_pred_boxes,
                targets=targets,
            )
            target_box = dynamic_targets["target_box"]
            target_obj = dynamic_targets["target_obj"]
            target_cls = dynamic_targets["target_cls"]
            object_mask = dynamic_targets["object_mask"]
            noobj_mask = dynamic_targets["noobj_mask"]
            collision_count = dynamic_targets["collision_count"]
            ignored_count = dynamic_targets["ignored_count"]
            dropped_gt_count = dynamic_targets["dropped_gt_count"]
        else:
            target_box = targets["target_box"]
            target_obj = targets["target_obj"]
            target_cls = targets["target_cls"]
            object_mask = targets["object_mask"]
            noobj_mask = targets["noobj_mask"]
            collision_count = targets["collision_count"]
            ignored_count = targets["ignored_count"]
            dropped_gt_count = targets["dropped_gt_count"]

        num_pos = object_mask.sum().clamp(min=1.0)
        num_neg = noobj_mask.sum().clamp(min=1.0)

        decoded_target_boxes = target_boxes_to_xyxy(target_box)
        giou = generalized_box_iou(decoded_pred_boxes, decoded_target_boxes)
        iou = box_iou_xyxy(decoded_pred_boxes, decoded_target_boxes)

        positive_mask = object_mask > 0
        if positive_mask.any():
            loss_box = (1.0 - giou[positive_mask]).mean()
            mean_giou = giou[positive_mask].mean()
        else:
            zero = pred_box.sum() * 0.0
            loss_box = zero
            mean_giou = zero

        if self.soft_objectness_target == "iou":
            # Keep a floor on positive obj targets so early bad boxes do not
            # collapse every positive sample into an almost-background target.
            positive_obj_target = iou.detach().clamp(min=self.soft_objectness_min, max=1.0)
        else:
            positive_obj_target = target_obj

        if self.soft_classification_target == "iou":
            # Replace the one-hot positive target of the matched class with the
            # current box quality so low-IoU matches receive weaker cls reward.
            positive_cls_target = iou.detach().clamp(min=0.0, max=1.0)
            soft_target_cls = target_cls * positive_cls_target.unsqueeze(-1)
        else:
            soft_target_cls = target_cls

        loss_obj_pos = (self.bce(pred_obj, positive_obj_target) * object_mask).sum() / num_pos
        loss_obj_neg = (self.bce(pred_obj, target_obj) * noobj_mask).sum() / num_neg
        loss_obj = loss_obj_pos + self.lambda_noobj * loss_obj_neg

        cls_mask = object_mask.unsqueeze(-1)
        cls_bce = self.bce(pred_cls, soft_target_cls)
        if self.cls_loss_mode == "varifocal":
            pred_prob = pred_cls.sigmoid().detach()
            focal_weight = torch.where(
                soft_target_cls > 0,
                soft_target_cls,
                self.varifocal_alpha * pred_prob.pow(self.varifocal_gamma),
            )
            loss_cls = (cls_bce * focal_weight).sum() / (num_pos * self.num_classes)
        else:
            loss_cls = (cls_bce * cls_mask).sum() / (num_pos * self.num_classes)

        total_loss = (
            self.lambda_box * loss_box
            + self.lambda_obj * loss_obj
            + self.lambda_cls * loss_cls
        )

        positive_cells_per_image = object_mask.sum(dim=(1, 2)).mean()
        collision_count_mean = collision_count.float().mean()
        ignored_count_mean = ignored_count.float().mean()
        dropped_gt_count_mean = dropped_gt_count.float().mean()

        return {
            "total_loss": total_loss,
            "loss_box": loss_box,
            "loss_obj": loss_obj,
            "loss_cls": loss_cls,
            "loss_obj_pos": loss_obj_pos,
            "loss_obj_neg": loss_obj_neg,
            "mean_giou": mean_giou,
            "mean_obj_target": positive_obj_target[positive_mask].mean() if positive_mask.any() else pred_box.sum() * 0.0,
            "mean_cls_target": soft_target_cls[positive_mask].max(dim=-1).values.mean() if positive_mask.any() else pred_box.sum() * 0.0,
            "positive_cells_per_image": positive_cells_per_image,
            "collision_count": collision_count_mean,
            "ignored_count": ignored_count_mean,
            "dropped_gt_count": dropped_gt_count_mean,
        }

    def forward(self, pred, targets):
        """Return one loss dict for either single-scale or multi-scale predictions."""
        if isinstance(pred, dict):
            scale_results = []
            for scale_name, scale_pred in pred.items():
                scale_targets = dict(targets["multiscale_targets"][scale_name])
                scale_targets["boxes"] = targets["boxes"]
                scale_targets["labels"] = targets["labels"]
                scale_targets["resized_size"] = targets["resized_size"]
                scale_results.append(self._forward_single_scale(scale_pred, scale_targets, scale_name=scale_name))

            total_loss = scale_results[0]["total_loss"]
            for result in scale_results[1:]:
                total_loss = total_loss + result["total_loss"]

            mean_keys = {
                "mean_giou",
                "mean_obj_target",
                "mean_cls_target",
            }
            aggregated = {"total_loss": total_loss}
            for key in scale_results[0]:
                if key == "total_loss":
                    continue
                if key in mean_keys:
                    aggregated[key] = sum(result[key] for result in scale_results) / len(scale_results)
                else:
                    aggregated[key] = sum(result[key] for result in scale_results)
            return aggregated

        return self._forward_single_scale(pred, targets)
