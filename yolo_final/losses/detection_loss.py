"""Baseline detection loss kept simple until the full YOLO loss is introduced."""

import torch
import torch.nn as nn


class DetectionLoss(nn.Module):
    """Compute simple class and box losses on positive grid cells only."""

    def __init__(self, num_classes=80, lambda_cls=1.0, lambda_box=5.0):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_cls = lambda_cls
        self.lambda_box = lambda_box

    def forward(self, pred, targets):
        """Return baseline loss tensors in the same dict format as the fuller loss."""
        pred_cls = pred[..., : self.num_classes]
        pred_box = pred[..., self.num_classes :]
        target_cls = targets["target_cls"]
        target_box = targets["target_box"]
        object_mask = targets["object_mask"]
        collision_count = targets.get("collision_count", torch.zeros(pred.shape[0], device=pred.device))
        mask = object_mask.unsqueeze(-1).float()
        num_pos = object_mask.sum().clamp(min=1.0)

        cls_loss = (((pred_cls - target_cls) ** 2) * mask).sum() / (num_pos * self.num_classes)
        box_loss = (((pred_box - target_box) ** 2) * mask).sum() / (num_pos * 4)
        total_loss = self.lambda_cls * cls_loss + self.lambda_box * box_loss
        zero = total_loss.detach() * 0.0
        return {
            "total_loss": total_loss,
            "loss_box": box_loss,
            "loss_obj": zero,
            "loss_cls": cls_loss,
            "loss_obj_pos": zero,
            "loss_obj_neg": zero,
            "mean_giou": zero,
            "positive_cells_per_image": object_mask.sum(dim=(1, 2)).mean(),
            "collision_count": collision_count.float().mean(),
        }
