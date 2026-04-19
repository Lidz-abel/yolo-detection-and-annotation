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
        box_parameterization="legacy",
        soft_objectness_target="hard",
        soft_objectness_min=0.0,
        soft_classification_target="hard",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_box = lambda_box
        self.lambda_obj = lambda_obj
        self.lambda_noobj = lambda_noobj
        self.lambda_cls = lambda_cls
        self.num_boxes = num_boxes
        self.anchors = anchors or []
        self.box_parameterization = box_parameterization
        self.soft_objectness_target = soft_objectness_target
        self.soft_objectness_min = soft_objectness_min
        self.soft_classification_target = soft_classification_target
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, pred, targets):
        """Return a dict of loss tensors and monitoring metrics for one batch."""
        if self.num_boxes > 1:
            pred = pred.view(*pred.shape[:3], self.num_boxes, self.num_classes + 5)
        pred_box = pred[..., 0:4]
        pred_obj = pred[..., 4]
        pred_cls = pred[..., 5:]

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

        anchor_tensor = None
        if self.num_boxes > 1 and self.anchors:
            anchor_tensor = torch.tensor(self.anchors, dtype=pred_box.dtype, device=pred_box.device)
        decoded_pred_boxes = decode_box_predictions(
            pred_box,
            anchors=anchor_tensor,
            box_parameterization=self.box_parameterization,
        )
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
        loss_cls = (self.bce(pred_cls, soft_target_cls) * cls_mask).sum() / (num_pos * self.num_classes)

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
