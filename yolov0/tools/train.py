"""Formal baseline training entrypoint for yolov0 stage-two experiments."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from data.detection_dataset import DetectionDataset, detection_collate_fn
from engine.trainer import train_one_epoch, validate_one_epoch
from losses.detection_loss import DetectionLoss
from losses.yolo_loss import YOLOLoss
from models.detector import YOLOv0Baseline
from utils.config import load_config, parse_anchor_string, summarize_config
from utils.experiment import init_run, update_metadata, write_result_summary
from utils.modeling import count_parameters, describe_model_output
from utils.visualization import save_visualization_set


def parse_args():
    """Parse the config path and an optional runtime seed override."""
    parser = argparse.ArgumentParser(description="Train the yolov0 baseline detector.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "base_train.toml"),
        help="Path to the training config file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional runtime seed override.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=str,
        default="",
        help="Optional checkpoint path used to resume weights for the current run.",
    )
    parser.add_argument(
        "--resume-epoch",
        type=int,
        default=0,
        help="Last fully completed epoch in the source run when resuming.",
    )
    parser.add_argument(
        "--resume-best-epoch",
        type=int,
        default=0,
        help="Best epoch observed in the source run when resuming.",
    )
    parser.add_argument(
        "--resume-best-val-loss",
        type=float,
        default=float("inf"),
        help="Best validation loss observed in the source run when resuming.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Seed Python and torch RNGs for reproducible baseline runs."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_device(device_name: str):
    """Pick the requested device, or auto-select CUDA when available."""
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def build_loader(dataset, batch_size, shuffle, train_cfg):
    """Create one DataLoader with the configured worker and prefetch settings."""
    num_workers = int(train_cfg["num_workers"])
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=bool(train_cfg["pin_memory"]),
        persistent_workers=bool(train_cfg["persistent_workers"]) if num_workers > 0 else False,
        prefetch_factor=int(train_cfg["prefetch_factor"]) if num_workers > 0 else None,
        collate_fn=detection_collate_fn,
    )


def build_optimizer(model, train_cfg):
    """Build the requested optimizer from the baseline training config."""
    optimizer_name = str(train_cfg["optimizer"]).lower()
    lr = float(train_cfg["lr"])
    weight_decay = float(train_cfg["weight_decay"])

    if optimizer_name == "adamw":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if optimizer_name == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if optimizer_name == "sgd":
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def build_scheduler(optimizer, train_cfg):
    """Build the configured learning-rate scheduler for epoch-level stepping."""
    scheduler_name = str(train_cfg["scheduler"]).lower()
    epochs = int(train_cfg["epochs"])
    min_lr = float(train_cfg["min_lr"])

    if scheduler_name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=min_lr)
    if scheduler_name == "none":
        return None
    raise ValueError(f"Unsupported scheduler: {scheduler_name}")


def build_criterion(data_cfg, model_cfg, loss_cfg):
    """Build either the baseline loss or the fuller YOLO-style loss from config."""
    if bool(loss_cfg["use_objectness"]) or str(loss_cfg["iou_loss"]).lower() != "none":
        return YOLOLoss(
            num_classes=int(data_cfg["num_classes"]),
            lambda_box=float(loss_cfg["lambda_box"]),
            lambda_obj=float(loss_cfg.get("lambda_obj", 1.0)),
            lambda_noobj=float(loss_cfg.get("lambda_noobj", 0.5)),
            lambda_cls=float(loss_cfg["lambda_cls"]),
            num_boxes=int(model_cfg.get("num_boxes", 1)),
            anchors=parse_anchor_string(model_cfg.get("anchors")),
            box_parameterization=str(model_cfg.get("box_parameterization", "legacy")),
            soft_objectness_target=str(loss_cfg.get("soft_objectness_target", "hard")),
            soft_objectness_min=float(loss_cfg.get("soft_objectness_min", 0.0)),
            soft_classification_target=str(loss_cfg.get("soft_classification_target", "hard")),
            cls_loss_mode=str(loss_cfg.get("cls_loss_mode", "bce")),
            varifocal_alpha=float(loss_cfg.get("varifocal_alpha", 0.75)),
            varifocal_gamma=float(loss_cfg.get("varifocal_gamma", 2.0)),
            assignment_strategy=str(loss_cfg.get("assignment_strategy", "static")),
            dynamic_topk=int(loss_cfg.get("dynamic_topk", 2)),
            dynamic_center_radius=int(loss_cfg.get("dynamic_center_radius", 1)),
            dynamic_box_cost=float(loss_cfg.get("dynamic_box_cost", 3.0)),
            dynamic_cls_cost=float(loss_cfg.get("dynamic_cls_cost", 1.0)),
            dynamic_ignore_iou=float(loss_cfg.get("dynamic_ignore_iou", 0.5)),
        )
    return DetectionLoss(
        num_classes=int(data_cfg["num_classes"]),
        lambda_cls=float(loss_cfg["lambda_cls"]),
        lambda_box=float(loss_cfg["lambda_box"]),
    )


def maybe_wrap_model(model, train_cfg, device):
    """Wrap the model in DataParallel when multiple GPUs are available."""
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    use_data_parallel = bool(train_cfg["use_data_parallel"])
    if device.type == "cuda" and use_data_parallel and gpu_count > 1:
        return torch.nn.DataParallel(model), gpu_count
    return model, gpu_count


def get_state_dict(model):
    """Save the underlying module weights even when DataParallel is enabled."""
    if isinstance(model, torch.nn.DataParallel):
        return model.module.state_dict()
    return model.state_dict()


def load_state_dict(model, state_dict):
    """Load weights into the wrapped module or plain model uniformly."""
    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)


def main():
    """Train the first yolov0 baseline with config-driven logging and checkpoints."""
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)

    data_cfg = config["data"]
    model_cfg = config["model"]
    loss_cfg = config["loss"]
    train_cfg = config["train"]
    logging_cfg = config["logging"]
    evaluation_cfg = config["evaluation"]
    anchors = parse_anchor_string(model_cfg.get("anchors"))
    num_boxes = int(model_cfg.get("num_boxes", 1))

    if args.seed is not None:
        train_cfg["seed"] = args.seed

    stage_name = "full_loss_training" if bool(loss_cfg["use_objectness"]) else "baseline_training"
    set_seed(int(train_cfg["seed"]))
    device = build_device(str(train_cfg["device"]))
    run_info = init_run(PROJECT_ROOT, config_path, config)
    writer = SummaryWriter(log_dir=str(run_info["tensorboard_dir"]))

    update_metadata(
        run_info["metadata_path"],
        status="running",
        stage=stage_name,
        device=str(device),
    )

    train_dataset = DetectionDataset(
        manifest_path=data_cfg["train_manifest"],
        image_size=int(data_cfg["image_size"]),
        grid_size=int(data_cfg["grid_size"]),
        num_classes=int(data_cfg["num_classes"]),
        num_boxes=num_boxes,
        anchors=anchors,
        anchor_positive_iou=float(model_cfg.get("anchor_positive_iou", 0.25)),
        anchor_ignore_iou=float(model_cfg.get("anchor_ignore_iou", 0.5)),
        anchor_match_metric=str(model_cfg.get("anchor_match_metric", "iou")),
        anchor_shape_ratio=float(model_cfg.get("anchor_shape_ratio", 4.0)),
        anchor_ignore_shape_ratio=model_cfg.get("anchor_ignore_shape_ratio"),
        max_samples=int(data_cfg["train_max_samples"]),
    )
    val_dataset = DetectionDataset(
        manifest_path=data_cfg["val_manifest"],
        image_size=int(data_cfg["image_size"]),
        grid_size=int(data_cfg["grid_size"]),
        num_classes=int(data_cfg["num_classes"]),
        num_boxes=num_boxes,
        anchors=anchors,
        anchor_positive_iou=float(model_cfg.get("anchor_positive_iou", 0.25)),
        anchor_ignore_iou=float(model_cfg.get("anchor_ignore_iou", 0.5)),
        anchor_match_metric=str(model_cfg.get("anchor_match_metric", "iou")),
        anchor_shape_ratio=float(model_cfg.get("anchor_shape_ratio", 4.0)),
        anchor_ignore_shape_ratio=model_cfg.get("anchor_ignore_shape_ratio"),
        max_samples=int(data_cfg["val_max_samples"]),
    )
    train_loader = build_loader(train_dataset, int(train_cfg["batch_size"]), bool(train_cfg["shuffle"]), train_cfg)
    val_loader = build_loader(val_dataset, int(train_cfg["batch_size"]), False, train_cfg)

    model = YOLOv0Baseline(
        num_classes=int(data_cfg["num_classes"]),
        model_name=str(model_cfg["name"]),
        width_mult=float(model_cfg["width_mult"]),
        depth_mult=float(model_cfg["depth_mult"]),
        use_residual=bool(model_cfg["use_residual"]),
        num_boxes=num_boxes,
        head_type=str(model_cfg.get("head_type", "shared")),
    ).to(device)
    model, gpu_count = maybe_wrap_model(model, train_cfg, device)

    criterion = build_criterion(data_cfg, model_cfg, loss_cfg)
    optimizer = build_optimizer(model, train_cfg)
    scheduler = build_scheduler(optimizer, train_cfg)

    resume_checkpoint = Path(args.resume_checkpoint).resolve() if args.resume_checkpoint else None
    start_epoch = 1
    global_step = 0
    best_val_loss = float("inf")
    best_epoch = 0
    if resume_checkpoint is not None:
        state_dict = torch.load(resume_checkpoint, map_location=device)
        load_state_dict(model, state_dict)
        start_epoch = int(args.resume_epoch) + 1
        best_epoch = int(args.resume_best_epoch)
        best_val_loss = float(args.resume_best_val_loss)
        if scheduler is not None and int(args.resume_epoch) > 0:
            for _ in range(int(args.resume_epoch)):
                scheduler.step()

    param_stats = count_parameters(model)
    output_shape = describe_model_output(model, int(data_cfg["image_size"]), device)

    epochs = int(train_cfg["epochs"])
    max_steps_per_epoch = int(train_cfg["max_steps_per_epoch"])
    max_val_steps = int(train_cfg["max_val_steps"])
    log_every_steps = int(logging_cfg["log_every_steps"])
    val_interval_epochs = int(evaluation_cfg["val_interval_epochs"])
    save_interval_epochs = int(logging_cfg["save_interval_epochs"])
    visualization_cfg = config["visualization"]

    train_history: list[dict] = []
    val_history: list[dict] = []

    print("starting yolov0 training")
    print("run id:", run_info["run_id"])
    print("device:", device)
    print("train dataset length:", len(train_dataset))
    print("val dataset length:", len(val_dataset))
    print("model output shape:", output_shape)
    print("parameter total:", param_stats["total"])
    print("parameter trainable:", param_stats["trainable"])
    print("gpu count:", gpu_count)
    print("tensorboard dir:", run_info["tensorboard_dir"])
    print("output dir:", run_info["output_dir"])
    if resume_checkpoint is not None:
        print("resume checkpoint:", resume_checkpoint)
        print("resume start epoch:", start_epoch)
        print("resume best epoch:", best_epoch)
        print("resume best val loss:", best_val_loss)

    update_metadata(
        run_info["metadata_path"],
        resume_checkpoint=str(resume_checkpoint) if resume_checkpoint is not None else "",
        resume_epoch=int(args.resume_epoch),
        resume_best_epoch=best_epoch if resume_checkpoint is not None else 0,
        resume_best_val_loss=None if best_val_loss == float("inf") else best_val_loss,
    )

    try:
        for epoch_index in range(start_epoch, epochs + 1):
            train_metrics = train_one_epoch(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                loader=train_loader,
                device=device,
                epoch_index=epoch_index,
                writer=writer,
                global_step=global_step,
                log_every_steps=log_every_steps,
                max_steps_per_epoch=max_steps_per_epoch,
            )
            global_step = train_metrics["global_step"]
            train_history.append({"epoch": epoch_index, **train_metrics})

            print(
                f"[epoch {epoch_index:03d} train] "
                f"total = {train_metrics['total_loss']:.4f} | "
                f"box = {train_metrics['box_loss']:.4f} | "
                f"obj = {train_metrics['obj_loss']:.4f} | "
                f"cls = {train_metrics['cls_loss']:.4f} | "
                f"batches = {train_metrics['batch_count']} | "
                f"time = {train_metrics['duration_seconds']:.1f}s"
            )

            did_validate = False
            val_metrics = None
            if len(val_dataset) > 0 and (epoch_index % val_interval_epochs == 0 or epoch_index == epochs):
                val_metrics = validate_one_epoch(
                    model=model,
                    criterion=criterion,
                    loader=val_loader,
                    device=device,
                    epoch_index=epoch_index,
                    writer=writer,
                    max_val_steps=max_val_steps,
                )
                val_history.append({"epoch": epoch_index, **val_metrics})
                did_validate = True

                print(
                    f"[epoch {epoch_index:03d} val] "
                    f"total = {val_metrics['total_loss']:.4f} | "
                    f"box = {val_metrics['box_loss']:.4f} | "
                    f"obj = {val_metrics['obj_loss']:.4f} | "
                    f"cls = {val_metrics['cls_loss']:.4f} | "
                    f"batches = {val_metrics['batch_count']} | "
                    f"time = {val_metrics['duration_seconds']:.1f}s"
                )

                if val_metrics["total_loss"] < best_val_loss:
                    best_val_loss = val_metrics["total_loss"]
                    best_epoch = epoch_index
                    torch.save(get_state_dict(model), run_info["output_dir"] / "best.pth")

            torch.save(get_state_dict(model), run_info["output_dir"] / "last.pth")
            if epoch_index % save_interval_epochs == 0 or epoch_index == epochs:
                torch.save(get_state_dict(model), run_info["output_dir"] / f"epoch_{epoch_index:03d}.pth")

            if scheduler is not None:
                scheduler.step()

            if did_validate:
                update_metadata(
                    run_info["metadata_path"],
                    last_epoch=epoch_index,
                    best_epoch=best_epoch,
                    best_val_loss=best_val_loss,
                    global_step=global_step,
                )

        status = "completed"
    except KeyboardInterrupt:
        status = "interrupted"
        print("training interrupted by user")
    finally:
        writer.close()

    visualization_dir = run_info["output_dir"] / "visualizations"
    checkpoint_for_vis = run_info["output_dir"] / "best.pth"
    if not checkpoint_for_vis.exists():
        checkpoint_for_vis = run_info["output_dir"] / "last.pth"
    if checkpoint_for_vis.exists() and len(val_dataset) > 0:
        state_dict = torch.load(checkpoint_for_vis, map_location=device)
        load_state_dict(model, state_dict)
        save_visualization_set(
            model=model,
            dataset=val_dataset,
            output_dir=visualization_dir,
            device=device,
            num_classes=int(data_cfg["num_classes"]),
            num_boxes=num_boxes,
            anchors=anchors,
            box_parameterization=str(model_cfg.get("box_parameterization", "legacy")),
            max_samples=int(visualization_cfg.get("max_samples", 4)),
            score_threshold=float(visualization_cfg.get("score_threshold", 0.05)),
            top_k=int(visualization_cfg.get("top_k", 10)),
            score_alpha=float(visualization_cfg.get("score_alpha", 1.0)),
            score_beta=float(visualization_cfg.get("score_beta", 1.0)),
        )

    metadata = update_metadata(
        run_info["metadata_path"],
        status=status,
        stage=stage_name,
        last_epoch=train_history[-1]["epoch"] if train_history else 0,
        global_step=global_step,
        best_epoch=best_epoch,
        best_val_loss=None if best_val_loss == float("inf") else best_val_loss,
        model_output_shape=output_shape,
        parameter_total=param_stats["total"],
        parameter_trainable=param_stats["trainable"],
        visualization_dir=str(visualization_dir),
    )

    summary_lines = [
        f"run_id = {run_info['run_id']}",
        f"timestamp = {run_info['timestamp']}",
        f"status = {status}",
        f"stage = {stage_name}",
        f"project_root = {PROJECT_ROOT}",
        f"config_path = {config_path}",
        f"config_snapshot = {run_info['config_snapshot']}",
        f"tensorboard_dir = {run_info['tensorboard_dir']}",
        f"output_dir = {run_info['output_dir']}",
        f"record_dir = {run_info['record_dir']}",
        f"git_commit = {metadata['git_commit']}",
        f"device = {device}",
        f"gpu_count = {gpu_count}",
        f"resume_checkpoint = {resume_checkpoint if resume_checkpoint is not None else ''}",
        f"resume_epoch = {args.resume_epoch if resume_checkpoint is not None else 0}",
        f"train_dataset_length = {len(train_dataset)}",
        f"val_dataset_length = {len(val_dataset)}",
        f"model_output_shape = {output_shape}",
        f"parameter_total = {param_stats['total']}",
        f"parameter_trainable = {param_stats['trainable']}",
        f"visualization_dir = {visualization_dir}",
        "",
        "config summary:",
        *summarize_config(config),
        "",
        "train history:",
    ]

    for item in train_history:
        summary_lines.append(
            "epoch = {epoch:03d} | total = {total_loss:.6f} | box = {box_loss:.6f} | "
            "obj = {obj_loss:.6f} | cls = {cls_loss:.6f} | giou = {mean_giou:.6f} | "
            "obj_target = {mean_obj_target:.6f} | "
            "pos_cells = {positive_cells_per_image:.6f} | collisions = {collision_count:.6f} | "
            "ignored = {ignored_count:.6f} | dropped_gt = {dropped_gt_count:.6f} | "
            "batches = {batch_count} | time = {duration_seconds:.3f}s".format(**item)
        )

    summary_lines.extend(["", "val history:"])
    if val_history:
        for item in val_history:
            summary_lines.append(
                "epoch = {epoch:03d} | total = {total_loss:.6f} | box = {box_loss:.6f} | "
                "obj = {obj_loss:.6f} | cls = {cls_loss:.6f} | giou = {mean_giou:.6f} | "
                "obj_target = {mean_obj_target:.6f} | "
                "pos_cells = {positive_cells_per_image:.6f} | collisions = {collision_count:.6f} | "
                "ignored = {ignored_count:.6f} | dropped_gt = {dropped_gt_count:.6f} | "
                "batches = {batch_count} | time = {duration_seconds:.3f}s".format(**item)
            )
    else:
        summary_lines.append("validation was not triggered in this run.")

    summary_lines.extend(
        [
            "",
            "artifacts:",
            f"last_checkpoint = {run_info['output_dir'] / 'last.pth'}",
            f"best_checkpoint = {run_info['output_dir'] / 'best.pth'}",
            f"best_epoch = {best_epoch}",
            f"best_val_loss = {None if best_val_loss == float('inf') else best_val_loss}",
        ]
    )

    write_result_summary(run_info["result_path"], summary_lines)
    print("training status:", status)
    print("result summary:", run_info["result_path"])


if __name__ == "__main__":
    main()
