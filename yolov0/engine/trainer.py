"""Epoch-level train and validation loops for the yolov0 baseline."""

from __future__ import annotations

import time

import torch


def _move_batch_to_device(images, targets, device):
    """Move the tensor parts of one batch onto the selected device."""
    moved_targets = {}
    for key, value in targets.items():
        if torch.is_tensor(value):
            moved_targets[key] = value.to(device, non_blocking=True)
        else:
            moved_targets[key] = value
    return images.to(device, non_blocking=True), moved_targets


def train_one_epoch(
    model,
    criterion,
    optimizer,
    loader,
    device,
    epoch_index,
    writer,
    global_step,
    log_every_steps,
    max_steps_per_epoch=0,
):
    """Run one training epoch and log step/epoch losses into TensorBoard."""
    model.train()

    total_loss_sum = 0.0
    box_loss_sum = 0.0
    obj_loss_sum = 0.0
    cls_loss_sum = 0.0
    obj_pos_loss_sum = 0.0
    obj_neg_loss_sum = 0.0
    mean_giou_sum = 0.0
    mean_obj_target_sum = 0.0
    mean_cls_target_sum = 0.0
    positive_cells_sum = 0.0
    collision_count_sum = 0.0
    ignored_count_sum = 0.0
    dropped_gt_count_sum = 0.0
    batch_count = 0
    start_time = time.perf_counter()

    for batch_index, (images, targets) in enumerate(loader, start=1):
        if max_steps_per_epoch and batch_index > max_steps_per_epoch:
            break

        images, targets = _move_batch_to_device(images, targets, device)

        optimizer.zero_grad(set_to_none=True)
        pred = model(images)
        loss_dict = criterion(pred, targets)
        total_loss = loss_dict["total_loss"]
        total_loss.backward()
        optimizer.step()

        global_step += 1
        batch_count += 1
        total_loss_sum += float(total_loss.item())
        box_loss_sum += float(loss_dict["loss_box"].item())
        obj_loss_sum += float(loss_dict["loss_obj"].item())
        cls_loss_sum += float(loss_dict["loss_cls"].item())
        obj_pos_loss_sum += float(loss_dict["loss_obj_pos"].item())
        obj_neg_loss_sum += float(loss_dict["loss_obj_neg"].item())
        mean_giou_sum += float(loss_dict["mean_giou"].item())
        mean_obj_target_sum += float(loss_dict["mean_obj_target"].item())
        mean_cls_target_sum += float(loss_dict["mean_cls_target"].item())
        positive_cells_sum += float(loss_dict["positive_cells_per_image"].item())
        collision_count_sum += float(loss_dict["collision_count"].item())
        ignored_count_sum += float(loss_dict["ignored_count"].item())
        dropped_gt_count_sum += float(loss_dict["dropped_gt_count"].item())

        writer.add_scalar("loss/total_step", total_loss.item(), global_step)
        writer.add_scalar("loss/box_step", loss_dict["loss_box"].item(), global_step)
        writer.add_scalar("loss/objectness_step", loss_dict["loss_obj"].item(), global_step)
        writer.add_scalar("loss/classification_step", loss_dict["loss_cls"].item(), global_step)
        writer.add_scalar("loss/objectness_positive_step", loss_dict["loss_obj_pos"].item(), global_step)
        writer.add_scalar("loss/objectness_negative_step", loss_dict["loss_obj_neg"].item(), global_step)
        writer.add_scalar("metrics/mean_giou_step", loss_dict["mean_giou"].item(), global_step)
        writer.add_scalar("metrics/mean_obj_target_step", loss_dict["mean_obj_target"].item(), global_step)
        writer.add_scalar("metrics/mean_cls_target_step", loss_dict["mean_cls_target"].item(), global_step)
        writer.add_scalar("metrics/positive_cells_per_image_step", loss_dict["positive_cells_per_image"].item(), global_step)
        writer.add_scalar("metrics/collision_count_step", loss_dict["collision_count"].item(), global_step)
        writer.add_scalar("metrics/ignored_count_step", loss_dict["ignored_count"].item(), global_step)
        writer.add_scalar("metrics/dropped_gt_count_step", loss_dict["dropped_gt_count"].item(), global_step)
        writer.add_scalar("train/lr_step", optimizer.param_groups[0]["lr"], global_step)

        if batch_index == 1 or batch_index % log_every_steps == 0:
            print(
                f"epoch {epoch_index:03d} | step {batch_index:04d} | "
                f"total = {total_loss.item():.4f} | "
                f"box = {loss_dict['loss_box'].item():.4f} | "
                f"obj = {loss_dict['loss_obj'].item():.4f} | "
                f"cls = {loss_dict['loss_cls'].item():.4f} | "
                f"lr = {optimizer.param_groups[0]['lr']:.6f}"
            )

    duration = time.perf_counter() - start_time
    if batch_count == 0:
        return {
            "total_loss": 0.0,
            "box_loss": 0.0,
            "obj_loss": 0.0,
            "cls_loss": 0.0,
            "obj_pos_loss": 0.0,
            "obj_neg_loss": 0.0,
            "mean_giou": 0.0,
            "mean_obj_target": 0.0,
            "mean_cls_target": 0.0,
            "positive_cells_per_image": 0.0,
            "collision_count": 0.0,
            "ignored_count": 0.0,
            "dropped_gt_count": 0.0,
            "batch_count": 0,
            "duration_seconds": duration,
            "global_step": global_step,
        }

    total_loss_mean = total_loss_sum / batch_count
    box_loss_mean = box_loss_sum / batch_count
    obj_loss_mean = obj_loss_sum / batch_count
    cls_loss_mean = cls_loss_sum / batch_count
    obj_pos_loss_mean = obj_pos_loss_sum / batch_count
    obj_neg_loss_mean = obj_neg_loss_sum / batch_count
    mean_giou_mean = mean_giou_sum / batch_count
    mean_obj_target_mean = mean_obj_target_sum / batch_count
    mean_cls_target_mean = mean_cls_target_sum / batch_count
    positive_cells_mean = positive_cells_sum / batch_count
    collision_count_mean = collision_count_sum / batch_count
    ignored_count_mean = ignored_count_sum / batch_count
    dropped_gt_count_mean = dropped_gt_count_sum / batch_count

    writer.add_scalar("loss/total_epoch", total_loss_mean, epoch_index)
    writer.add_scalar("loss/box_epoch", box_loss_mean, epoch_index)
    writer.add_scalar("loss/objectness_epoch", obj_loss_mean, epoch_index)
    writer.add_scalar("loss/classification_epoch", cls_loss_mean, epoch_index)
    writer.add_scalar("loss/objectness_positive_epoch", obj_pos_loss_mean, epoch_index)
    writer.add_scalar("loss/objectness_negative_epoch", obj_neg_loss_mean, epoch_index)
    writer.add_scalar("metrics/mean_giou_epoch", mean_giou_mean, epoch_index)
    writer.add_scalar("metrics/mean_obj_target_epoch", mean_obj_target_mean, epoch_index)
    writer.add_scalar("metrics/mean_cls_target_epoch", mean_cls_target_mean, epoch_index)
    writer.add_scalar("metrics/positive_cells_per_image_epoch", positive_cells_mean, epoch_index)
    writer.add_scalar("metrics/collision_count_epoch", collision_count_mean, epoch_index)
    writer.add_scalar("metrics/ignored_count_epoch", ignored_count_mean, epoch_index)
    writer.add_scalar("metrics/dropped_gt_count_epoch", dropped_gt_count_mean, epoch_index)
    writer.add_scalar("train/lr_epoch", optimizer.param_groups[0]["lr"], epoch_index)

    return {
        "total_loss": total_loss_mean,
        "box_loss": box_loss_mean,
        "obj_loss": obj_loss_mean,
        "cls_loss": cls_loss_mean,
        "obj_pos_loss": obj_pos_loss_mean,
        "obj_neg_loss": obj_neg_loss_mean,
        "mean_giou": mean_giou_mean,
        "mean_obj_target": mean_obj_target_mean,
        "mean_cls_target": mean_cls_target_mean,
        "positive_cells_per_image": positive_cells_mean,
        "collision_count": collision_count_mean,
        "ignored_count": ignored_count_mean,
        "dropped_gt_count": dropped_gt_count_mean,
        "batch_count": batch_count,
        "duration_seconds": duration,
        "global_step": global_step,
    }


def validate_one_epoch(
    model,
    criterion,
    loader,
    device,
    epoch_index,
    writer,
    max_val_steps=0,
):
    """Run one validation pass and log averaged validation losses."""
    model.eval()

    total_loss_sum = 0.0
    box_loss_sum = 0.0
    obj_loss_sum = 0.0
    cls_loss_sum = 0.0
    obj_pos_loss_sum = 0.0
    obj_neg_loss_sum = 0.0
    mean_giou_sum = 0.0
    mean_obj_target_sum = 0.0
    mean_cls_target_sum = 0.0
    positive_cells_sum = 0.0
    collision_count_sum = 0.0
    ignored_count_sum = 0.0
    dropped_gt_count_sum = 0.0
    batch_count = 0
    start_time = time.perf_counter()

    with torch.no_grad():
        for batch_index, (images, targets) in enumerate(loader, start=1):
            if max_val_steps and batch_index > max_val_steps:
                break

            images, targets = _move_batch_to_device(images, targets, device)
            pred = model(images)
            loss_dict = criterion(pred, targets)
            total_loss = loss_dict["total_loss"]

            batch_count += 1
            total_loss_sum += float(total_loss.item())
            box_loss_sum += float(loss_dict["loss_box"].item())
            obj_loss_sum += float(loss_dict["loss_obj"].item())
            cls_loss_sum += float(loss_dict["loss_cls"].item())
            obj_pos_loss_sum += float(loss_dict["loss_obj_pos"].item())
            obj_neg_loss_sum += float(loss_dict["loss_obj_neg"].item())
            mean_giou_sum += float(loss_dict["mean_giou"].item())
            mean_obj_target_sum += float(loss_dict["mean_obj_target"].item())
            mean_cls_target_sum += float(loss_dict["mean_cls_target"].item())
            positive_cells_sum += float(loss_dict["positive_cells_per_image"].item())
            collision_count_sum += float(loss_dict["collision_count"].item())
            ignored_count_sum += float(loss_dict["ignored_count"].item())
            dropped_gt_count_sum += float(loss_dict["dropped_gt_count"].item())

    duration = time.perf_counter() - start_time
    if batch_count == 0:
        metrics = {
            "total_loss": 0.0,
            "box_loss": 0.0,
            "obj_loss": 0.0,
            "cls_loss": 0.0,
            "obj_pos_loss": 0.0,
            "obj_neg_loss": 0.0,
            "mean_giou": 0.0,
            "mean_obj_target": 0.0,
            "mean_cls_target": 0.0,
            "positive_cells_per_image": 0.0,
            "collision_count": 0.0,
            "ignored_count": 0.0,
            "dropped_gt_count": 0.0,
            "batch_count": 0,
            "duration_seconds": duration,
        }
    else:
        metrics = {
            "total_loss": total_loss_sum / batch_count,
            "box_loss": box_loss_sum / batch_count,
            "obj_loss": obj_loss_sum / batch_count,
            "cls_loss": cls_loss_sum / batch_count,
            "obj_pos_loss": obj_pos_loss_sum / batch_count,
            "obj_neg_loss": obj_neg_loss_sum / batch_count,
            "mean_giou": mean_giou_sum / batch_count,
            "mean_obj_target": mean_obj_target_sum / batch_count,
            "mean_cls_target": mean_cls_target_sum / batch_count,
            "positive_cells_per_image": positive_cells_sum / batch_count,
            "collision_count": collision_count_sum / batch_count,
            "ignored_count": ignored_count_sum / batch_count,
            "dropped_gt_count": dropped_gt_count_sum / batch_count,
            "batch_count": batch_count,
            "duration_seconds": duration,
        }

    writer.add_scalar("val/total_epoch", metrics["total_loss"], epoch_index)
    writer.add_scalar("val/box_epoch", metrics["box_loss"], epoch_index)
    writer.add_scalar("val/objectness_epoch", metrics["obj_loss"], epoch_index)
    writer.add_scalar("val/classification_epoch", metrics["cls_loss"], epoch_index)
    writer.add_scalar("val/objectness_positive_epoch", metrics["obj_pos_loss"], epoch_index)
    writer.add_scalar("val/objectness_negative_epoch", metrics["obj_neg_loss"], epoch_index)
    writer.add_scalar("val/mean_giou_epoch", metrics["mean_giou"], epoch_index)
    writer.add_scalar("val/mean_obj_target_epoch", metrics["mean_obj_target"], epoch_index)
    writer.add_scalar("val/mean_cls_target_epoch", metrics["mean_cls_target"], epoch_index)
    writer.add_scalar("val/positive_cells_per_image_epoch", metrics["positive_cells_per_image"], epoch_index)
    writer.add_scalar("val/collision_count_epoch", metrics["collision_count"], epoch_index)
    writer.add_scalar("val/ignored_count_epoch", metrics["ignored_count"], epoch_index)
    writer.add_scalar("val/dropped_gt_count_epoch", metrics["dropped_gt_count"], epoch_index)
    return metrics
