"""Distributed train and validation loops for DDP detector experiments."""

from __future__ import annotations

import time

import torch
import torch.distributed as dist

from engine.trainer import _move_batch_to_device


METRIC_KEYS = (
    "total_loss",
    "box_loss",
    "obj_loss",
    "cls_loss",
    "obj_pos_loss",
    "obj_neg_loss",
    "mean_giou",
    "mean_obj_target",
    "mean_cls_target",
    "positive_cells_per_image",
    "collision_count",
    "ignored_count",
    "dropped_gt_count",
)


LOSS_TO_METRIC = {
    "total_loss": "total_loss",
    "loss_box": "box_loss",
    "loss_obj": "obj_loss",
    "loss_cls": "cls_loss",
    "loss_obj_pos": "obj_pos_loss",
    "loss_obj_neg": "obj_neg_loss",
    "mean_giou": "mean_giou",
    "mean_obj_target": "mean_obj_target",
    "mean_cls_target": "mean_cls_target",
    "positive_cells_per_image": "positive_cells_per_image",
    "collision_count": "collision_count",
    "ignored_count": "ignored_count",
    "dropped_gt_count": "dropped_gt_count",
}


def _is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def _all_reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    if _is_distributed():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def _all_reduce_max(tensor: torch.Tensor) -> torch.Tensor:
    if _is_distributed():
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return tensor


def _empty_metrics(duration: float, global_step: int | None = None) -> dict:
    metrics = {key: 0.0 for key in METRIC_KEYS}
    metrics.update(
        {
            "batch_count": 0,
            "optimizer_steps": 0,
            "global_batch_count": 0,
            "sample_count": 0,
            "duration_seconds": duration,
        }
    )
    if global_step is not None:
        metrics["global_step"] = global_step
    return metrics


def _aggregate_epoch_metrics(
    metric_sums: dict[str, float],
    local_batch_count: int,
    sample_count: int,
    duration: float,
    device,
) -> dict:
    values = [metric_sums[key] for key in METRIC_KEYS]
    values.extend([float(sample_count), float(local_batch_count)])
    stat_tensor = torch.tensor(values, dtype=torch.float64, device=device)
    _all_reduce_sum(stat_tensor)

    step_tensor = torch.tensor([float(local_batch_count)], dtype=torch.float64, device=device)
    _all_reduce_max(step_tensor)

    duration_tensor = torch.tensor([duration], dtype=torch.float64, device=device)
    _all_reduce_max(duration_tensor)

    global_sample_count = int(stat_tensor[-2].item())
    global_batch_count = int(stat_tensor[-1].item())
    optimizer_steps = int(step_tensor.item())
    if global_sample_count == 0:
        return _empty_metrics(float(duration_tensor.item()))

    metrics = {
        key: float(stat_tensor[index].item() / global_sample_count)
        for index, key in enumerate(METRIC_KEYS)
    }
    metrics["batch_count"] = optimizer_steps
    metrics["optimizer_steps"] = optimizer_steps
    metrics["global_batch_count"] = global_batch_count
    metrics["sample_count"] = global_sample_count
    metrics["duration_seconds"] = float(duration_tensor.item())
    return metrics


def _write_step_scalars(writer, loss_dict, optimizer, global_step: int) -> None:
    if writer is None:
        return
    writer.add_scalar("rank0_loss/total_step", loss_dict["total_loss"].item(), global_step)
    writer.add_scalar("rank0_loss/box_step", loss_dict["loss_box"].item(), global_step)
    writer.add_scalar("rank0_loss/objectness_step", loss_dict["loss_obj"].item(), global_step)
    writer.add_scalar("rank0_loss/classification_step", loss_dict["loss_cls"].item(), global_step)
    writer.add_scalar("rank0_loss/objectness_positive_step", loss_dict["loss_obj_pos"].item(), global_step)
    writer.add_scalar("rank0_loss/objectness_negative_step", loss_dict["loss_obj_neg"].item(), global_step)
    writer.add_scalar("rank0_metrics/mean_giou_step", loss_dict["mean_giou"].item(), global_step)
    writer.add_scalar("rank0_metrics/mean_obj_target_step", loss_dict["mean_obj_target"].item(), global_step)
    writer.add_scalar("rank0_metrics/mean_cls_target_step", loss_dict["mean_cls_target"].item(), global_step)
    writer.add_scalar(
        "rank0_metrics/positive_cells_per_image_step",
        loss_dict["positive_cells_per_image"].item(),
        global_step,
    )
    writer.add_scalar("rank0_metrics/collision_count_step", loss_dict["collision_count"].item(), global_step)
    writer.add_scalar("rank0_metrics/ignored_count_step", loss_dict["ignored_count"].item(), global_step)
    writer.add_scalar("rank0_metrics/dropped_gt_count_step", loss_dict["dropped_gt_count"].item(), global_step)
    writer.add_scalar("train/lr_step", optimizer.param_groups[0]["lr"], global_step)


def _write_epoch_scalars(writer, prefix: str, metrics: dict, epoch_index: int, optimizer=None) -> None:
    if writer is None:
        return
    if prefix == "loss":
        writer.add_scalar("loss/total_epoch", metrics["total_loss"], epoch_index)
        writer.add_scalar("loss/box_epoch", metrics["box_loss"], epoch_index)
        writer.add_scalar("loss/objectness_epoch", metrics["obj_loss"], epoch_index)
        writer.add_scalar("loss/classification_epoch", metrics["cls_loss"], epoch_index)
        writer.add_scalar("loss/objectness_positive_epoch", metrics["obj_pos_loss"], epoch_index)
        writer.add_scalar("loss/objectness_negative_epoch", metrics["obj_neg_loss"], epoch_index)
        writer.add_scalar("metrics/mean_giou_epoch", metrics["mean_giou"], epoch_index)
        writer.add_scalar("metrics/mean_obj_target_epoch", metrics["mean_obj_target"], epoch_index)
        writer.add_scalar("metrics/mean_cls_target_epoch", metrics["mean_cls_target"], epoch_index)
        writer.add_scalar("metrics/positive_cells_per_image_epoch", metrics["positive_cells_per_image"], epoch_index)
        writer.add_scalar("metrics/collision_count_epoch", metrics["collision_count"], epoch_index)
        writer.add_scalar("metrics/ignored_count_epoch", metrics["ignored_count"], epoch_index)
        writer.add_scalar("metrics/dropped_gt_count_epoch", metrics["dropped_gt_count"], epoch_index)
        if optimizer is not None:
            writer.add_scalar("train/lr_epoch", optimizer.param_groups[0]["lr"], epoch_index)
        return

    writer.add_scalar(f"{prefix}/total_epoch", metrics["total_loss"], epoch_index)
    writer.add_scalar(f"{prefix}/box_epoch", metrics["box_loss"], epoch_index)
    writer.add_scalar(f"{prefix}/objectness_epoch", metrics["obj_loss"], epoch_index)
    writer.add_scalar(f"{prefix}/classification_epoch", metrics["cls_loss"], epoch_index)
    writer.add_scalar(f"{prefix}/objectness_positive_epoch", metrics["obj_pos_loss"], epoch_index)
    writer.add_scalar(f"{prefix}/objectness_negative_epoch", metrics["obj_neg_loss"], epoch_index)
    writer.add_scalar(f"{prefix}/mean_giou_epoch", metrics["mean_giou"], epoch_index)
    writer.add_scalar(f"{prefix}/mean_obj_target_epoch", metrics["mean_obj_target"], epoch_index)
    writer.add_scalar(f"{prefix}/mean_cls_target_epoch", metrics["mean_cls_target"], epoch_index)
    writer.add_scalar(f"{prefix}/positive_cells_per_image_epoch", metrics["positive_cells_per_image"], epoch_index)
    writer.add_scalar(f"{prefix}/collision_count_epoch", metrics["collision_count"], epoch_index)
    writer.add_scalar(f"{prefix}/ignored_count_epoch", metrics["ignored_count"], epoch_index)
    writer.add_scalar(f"{prefix}/dropped_gt_count_epoch", metrics["dropped_gt_count"], epoch_index)
    if optimizer is not None:
        writer.add_scalar("train/lr_epoch", optimizer.param_groups[0]["lr"], epoch_index)


def train_one_epoch_ddp(
    model,
    criterion,
    optimizer,
    loader,
    device,
    epoch_index,
    writer,
    global_step,
    log_every_steps,
    rank,
    max_steps_per_epoch=0,
):
    """Run one DDP training epoch and return globally averaged metrics."""
    model.train()

    metric_sums = {key: 0.0 for key in METRIC_KEYS}
    batch_count = 0
    sample_count = 0
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

        batch_samples = int(images.shape[0])
        global_step += 1
        batch_count += 1
        sample_count += batch_samples
        for loss_key, metric_key in LOSS_TO_METRIC.items():
            metric_sums[metric_key] += float(loss_dict[loss_key].item()) * batch_samples

        if rank == 0:
            _write_step_scalars(writer, loss_dict, optimizer, global_step)
            if batch_index == 1 or batch_index % log_every_steps == 0:
                print(
                    f"epoch {epoch_index:03d} | step {batch_index:04d} | "
                    f"total = {total_loss.item():.4f} | "
                    f"box = {loss_dict['loss_box'].item():.4f} | "
                    f"obj = {loss_dict['loss_obj'].item():.4f} | "
                    f"cls = {loss_dict['loss_cls'].item():.4f} | "
                    f"lr = {optimizer.param_groups[0]['lr']:.6f}",
                    flush=True,
                )

    duration = time.perf_counter() - start_time
    metrics = _aggregate_epoch_metrics(metric_sums, batch_count, sample_count, duration, device)
    metrics["global_step"] = global_step
    if rank == 0:
        _write_epoch_scalars(writer, "loss", metrics, epoch_index, optimizer=optimizer)
    return metrics


def validate_one_epoch_ddp(
    model,
    criterion,
    loader,
    device,
    epoch_index,
    writer,
    rank,
    max_val_steps=0,
):
    """Run one DDP validation pass and return globally averaged metrics."""
    model.eval()

    metric_sums = {key: 0.0 for key in METRIC_KEYS}
    batch_count = 0
    sample_count = 0
    start_time = time.perf_counter()

    with torch.no_grad():
        for batch_index, (images, targets) in enumerate(loader, start=1):
            if max_val_steps and batch_index > max_val_steps:
                break

            images, targets = _move_batch_to_device(images, targets, device)
            pred = model(images)
            loss_dict = criterion(pred, targets)

            batch_samples = int(images.shape[0])
            batch_count += 1
            sample_count += batch_samples
            for loss_key, metric_key in LOSS_TO_METRIC.items():
                metric_sums[metric_key] += float(loss_dict[loss_key].item()) * batch_samples

    duration = time.perf_counter() - start_time
    metrics = _aggregate_epoch_metrics(metric_sums, batch_count, sample_count, duration, device)
    if rank == 0:
        _write_epoch_scalars(writer, "val", metrics, epoch_index)
    return metrics
