"""DDP training entrypoint for yolov0 experiments.

Launch example:
    torchrun --standalone --nproc_per_node=8 tools/train_ddp.py --config configs/dual_scale_three_box_coco_only_noobj1.toml
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from data.detection_dataset import DetectionDataset, detection_collate_fn
from engine.distributed_trainer import train_one_epoch_ddp, validate_one_epoch_ddp
from models.detector import YOLOv0Baseline
from train import (
    build_criterion,
    build_dataset_storage_kwargs,
    build_optimizer,
    build_scheduler,
    load_state_dict,
    set_seed,
    validate_configured_storage,
)
from utils.config import (
    load_config,
    parse_anchor_map,
    parse_anchor_string,
    parse_int_list,
    parse_string_list,
    summarize_config,
)
from utils.experiment import init_run, update_metadata, write_result_summary
from utils.modeling import count_parameters, describe_model_output
from utils.visualization import save_visualization_set


def parse_args():
    parser = argparse.ArgumentParser(description="Train yolov0 with DistributedDataParallel.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "base_train.toml"),
        help="Path to the training config file.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional runtime seed override.")
    parser.add_argument("--resume-checkpoint", type=str, default="", help="Optional checkpoint path.")
    parser.add_argument("--resume-epoch", type=int, default=0, help="Last fully completed resume epoch.")
    parser.add_argument("--resume-best-epoch", type=int, default=0, help="Best epoch in the source run.")
    parser.add_argument(
        "--resume-best-val-loss",
        type=float,
        default=float("inf"),
        help="Best validation loss in the source run.",
    )
    parser.add_argument(
        "--batch-size-per-rank",
        action="store_true",
        help="Interpret config train.batch_size as per-rank batch size. Default treats it as global batch size.",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Runtime override for train.batch_size.")
    parser.add_argument("--epochs", type=int, default=None, help="Runtime override for train.epochs.")
    parser.add_argument("--num-workers", type=int, default=None, help="Runtime override for train.num_workers.")
    parser.add_argument(
        "--max-steps-per-epoch",
        type=int,
        default=None,
        help="Runtime override for train.max_steps_per_epoch.",
    )
    parser.add_argument("--max-val-steps", type=int, default=None, help="Runtime override for train.max_val_steps.")
    parser.add_argument(
        "--run-name-suffix",
        type=str,
        default="ddp",
        help="Suffix appended to logging.run_name so DDP runs are easy to distinguish.",
    )
    return parser.parse_args()


def setup_distributed():
    """Initialize torch distributed from torchrun environment variables."""
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    has_cuda = torch.cuda.is_available()
    backend = "nccl" if has_cuda else "gloo"
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")

    if has_cuda:
        if local_rank >= torch.cuda.device_count():
            raise RuntimeError(
                f"LOCAL_RANK={local_rank} but only {torch.cuda.device_count()} CUDA devices are visible."
            )
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    return rank, local_rank, world_size, device, backend


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_rank0(rank: int) -> bool:
    return rank == 0


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def broadcast_object(value, src: int = 0):
    if not (dist.is_available() and dist.is_initialized()):
        return value
    payload = [value]
    dist.broadcast_object_list(payload, src=src)
    return payload[0]


def apply_runtime_overrides(config: dict, args, rank: int) -> None:
    train_cfg = config["train"]
    logging_cfg = config["logging"]

    if args.seed is not None:
        train_cfg["seed"] = args.seed
    if args.batch_size is not None:
        train_cfg["batch_size"] = args.batch_size
    if args.epochs is not None:
        train_cfg["epochs"] = args.epochs
    if args.num_workers is not None:
        train_cfg["num_workers"] = args.num_workers
    if args.max_steps_per_epoch is not None:
        train_cfg["max_steps_per_epoch"] = args.max_steps_per_epoch
    if args.max_val_steps is not None:
        train_cfg["max_val_steps"] = args.max_val_steps

    suffix = str(args.run_name_suffix).strip()
    if suffix:
        run_name = str(logging_cfg["run_name"])
        if not run_name.endswith(f"_{suffix}"):
            logging_cfg["run_name"] = f"{run_name}_{suffix}"

    if rank == 0 and bool(train_cfg.get("use_data_parallel", False)):
        print("DDP mode: ignoring train.use_data_parallel from config.", flush=True)


def resolve_per_rank_settings(train_cfg: dict, world_size: int, args, rank: int) -> tuple[int, int]:
    configured_batch_size = int(train_cfg["batch_size"])
    configured_workers = int(train_cfg["num_workers"])

    if args.batch_size_per_rank:
        per_rank_batch_size = configured_batch_size
        effective_global_batch_size = configured_batch_size * world_size
    else:
        per_rank_batch_size = max(1, configured_batch_size // world_size)
        effective_global_batch_size = per_rank_batch_size * world_size
        if configured_batch_size % world_size != 0 and rank == 0:
            print(
                "DDP warning: train.batch_size is not divisible by world_size; "
                f"using per-rank batch {per_rank_batch_size}, effective global batch {effective_global_batch_size}.",
                flush=True,
            )

    per_rank_workers = max(0, math.ceil(configured_workers / max(world_size, 1)))
    if rank == 0:
        print("DDP configured global batch size:", configured_batch_size, flush=True)
        print("DDP per-rank batch size:", per_rank_batch_size, flush=True)
        print("DDP effective global batch size:", effective_global_batch_size, flush=True)
        print("DDP configured workers:", configured_workers, flush=True)
        print("DDP per-rank workers:", per_rank_workers, flush=True)
    return per_rank_batch_size, per_rank_workers


def build_distributed_loader(dataset, batch_size, shuffle, train_cfg, rank, world_size, num_workers):
    def seed_worker(worker_id):
        worker_seed = (torch.initial_seed() + rank * 100000 + worker_id) % 2**32
        random.seed(worker_seed)

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle,
        drop_last=False,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=bool(train_cfg["pin_memory"]),
        persistent_workers=bool(train_cfg["persistent_workers"]) if num_workers > 0 else False,
        prefetch_factor=int(train_cfg["prefetch_factor"]) if num_workers > 0 else None,
        collate_fn=detection_collate_fn,
        worker_init_fn=seed_worker if num_workers > 0 else None,
    ), sampler


def get_state_dict(model):
    if isinstance(model, DistributedDataParallel):
        return model.module.state_dict()
    return model.state_dict()


def save_checkpoint_ddp(model, checkpoint_path: Path) -> None:
    """Save a DDP checkpoint after ensuring the rank0 output directory exists."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(get_state_dict(model), checkpoint_path)


def load_state_dict_ddp(model, state_dict):
    if isinstance(model, DistributedDataParallel):
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)


def format_history_line(item: dict) -> str:
    return (
        "epoch = {epoch:03d} | total = {total_loss:.6f} | box = {box_loss:.6f} | "
        "obj = {obj_loss:.6f} | cls = {cls_loss:.6f} | giou = {mean_giou:.6f} | "
        "obj_target = {mean_obj_target:.6f} | "
        "pos_cells = {positive_cells_per_image:.6f} | collisions = {collision_count:.6f} | "
        "ignored = {ignored_count:.6f} | dropped_gt = {dropped_gt_count:.6f} | "
        "optimizer_steps = {optimizer_steps} | global_batches = {global_batch_count} | "
        "samples = {sample_count} | time = {duration_seconds:.3f}s"
    ).format(**item)


def main():
    args = parse_args()
    rank, local_rank, world_size, device, backend = setup_distributed()

    try:
        config_path = Path(args.config).resolve()
        config = load_config(config_path)
        apply_runtime_overrides(config, args, rank)

        data_cfg = config["data"]
        model_cfg = config["model"]
        loss_cfg = config["loss"]
        train_cfg = config["train"]
        logging_cfg = config["logging"]
        evaluation_cfg = config["evaluation"]
        visualization_cfg = config["visualization"]
        augmentation_cfg = config.get("augmentation", {})

        stage_name = "full_loss_training_ddp" if bool(loss_cfg["use_objectness"]) else "baseline_training_ddp"
        set_seed(int(train_cfg["seed"]))

        anchors = parse_anchor_string(model_cfg.get("anchors"))
        num_boxes = int(model_cfg.get("num_boxes", 1))
        grid_sizes = parse_int_list(data_cfg.get("grid_sizes"))
        feature_levels = parse_string_list(model_cfg.get("feature_levels"))
        if not grid_sizes:
            grid_sizes = [int(data_cfg["grid_size"])]
        if not feature_levels:
            feature_levels = [f"scale_{index}" for index in range(len(grid_sizes))]
        anchors_by_level = parse_anchor_map(model_cfg, feature_levels)
        default_grid_size = int(data_cfg.get("grid_size", grid_sizes[0]))

        if is_rank0(rank):
            validate_configured_storage(data_cfg)
            run_info = init_run(PROJECT_ROOT, config_path, config)
            update_metadata(
                run_info["metadata_path"],
                status="running",
                stage=stage_name,
                backend=backend,
                world_size=world_size,
                local_rank_count=world_size,
                device=str(device),
                launcher="torchrun",
            )
        else:
            run_info = None
        run_info = broadcast_object(run_info, src=0)
        barrier()

        dataset_storage_kwargs = build_dataset_storage_kwargs(data_cfg)
        train_dataset = DetectionDataset(
            manifest_path=data_cfg["train_manifest"],
            image_size=int(data_cfg["image_size"]),
            grid_size=default_grid_size,
            grid_sizes=grid_sizes,
            feature_levels=feature_levels,
            num_classes=int(data_cfg["num_classes"]),
            num_boxes=num_boxes,
            anchors=anchors,
            anchors_by_level=anchors_by_level,
            anchor_positive_iou=float(model_cfg.get("anchor_positive_iou", 0.25)),
            anchor_ignore_iou=float(model_cfg.get("anchor_ignore_iou", 0.5)),
            anchor_match_metric=str(model_cfg.get("anchor_match_metric", "iou")),
            anchor_shape_ratio=float(model_cfg.get("anchor_shape_ratio", 4.0)),
            anchor_ignore_shape_ratio=model_cfg.get("anchor_ignore_shape_ratio"),
            max_samples=int(data_cfg["train_max_samples"]),
            augmentation_cfg=augmentation_cfg if bool(augmentation_cfg.get("enabled", False)) else None,
            **dataset_storage_kwargs,
        )
        val_dataset = DetectionDataset(
            manifest_path=data_cfg["val_manifest"],
            image_size=int(data_cfg["image_size"]),
            grid_size=default_grid_size,
            grid_sizes=grid_sizes,
            feature_levels=feature_levels,
            num_classes=int(data_cfg["num_classes"]),
            num_boxes=num_boxes,
            anchors=anchors,
            anchors_by_level=anchors_by_level,
            anchor_positive_iou=float(model_cfg.get("anchor_positive_iou", 0.25)),
            anchor_ignore_iou=float(model_cfg.get("anchor_ignore_iou", 0.5)),
            anchor_match_metric=str(model_cfg.get("anchor_match_metric", "iou")),
            anchor_shape_ratio=float(model_cfg.get("anchor_shape_ratio", 4.0)),
            anchor_ignore_shape_ratio=model_cfg.get("anchor_ignore_shape_ratio"),
            max_samples=int(data_cfg["val_max_samples"]),
            **dataset_storage_kwargs,
        )

        per_rank_batch_size, per_rank_workers = resolve_per_rank_settings(train_cfg, world_size, args, rank)
        train_loader, train_sampler = build_distributed_loader(
            train_dataset,
            per_rank_batch_size,
            bool(train_cfg["shuffle"]),
            train_cfg,
            rank,
            world_size,
            per_rank_workers,
        )
        val_loader, val_sampler = build_distributed_loader(
            val_dataset,
            per_rank_batch_size,
            False,
            train_cfg,
            rank,
            world_size,
            per_rank_workers,
        )

        base_model = YOLOv0Baseline(
            num_classes=int(data_cfg["num_classes"]),
            model_name=str(model_cfg["name"]),
            width_mult=float(model_cfg["width_mult"]),
            depth_mult=float(model_cfg["depth_mult"]),
            use_residual=bool(model_cfg["use_residual"]),
            num_boxes=num_boxes,
            head_type=str(model_cfg.get("head_type", "shared")),
            neck_type=str(model_cfg.get("neck_type", "none")),
            feature_levels=feature_levels,
        ).to(device)

        resume_checkpoint = Path(args.resume_checkpoint).resolve() if args.resume_checkpoint else None
        start_epoch = 1
        global_step = 0
        best_val_loss = float("inf")
        best_epoch = 0
        if resume_checkpoint is not None:
            state_dict = torch.load(resume_checkpoint, map_location=device)
            load_state_dict(base_model, state_dict)
            start_epoch = int(args.resume_epoch) + 1
            best_epoch = int(args.resume_best_epoch)
            best_val_loss = float(args.resume_best_val_loss)

        param_stats = count_parameters(base_model)
        output_shape = describe_model_output(base_model, int(data_cfg["image_size"]), device)

        if world_size > 1:
            model = DistributedDataParallel(
                base_model,
                device_ids=[local_rank] if device.type == "cuda" else None,
                output_device=local_rank if device.type == "cuda" else None,
                find_unused_parameters=False,
            )
        else:
            model = base_model

        criterion = build_criterion(data_cfg, model_cfg, loss_cfg, feature_levels)
        optimizer = build_optimizer(model, train_cfg)
        scheduler = build_scheduler(optimizer, train_cfg)
        if scheduler is not None and int(args.resume_epoch) > 0:
            for _ in range(int(args.resume_epoch)):
                scheduler.step()

        writer = SummaryWriter(log_dir=str(run_info["tensorboard_dir"])) if is_rank0(rank) else None

        epochs = int(train_cfg["epochs"])
        max_steps_per_epoch = int(train_cfg["max_steps_per_epoch"])
        max_val_steps = int(train_cfg["max_val_steps"])
        log_every_steps = int(logging_cfg["log_every_steps"])
        val_interval_epochs = int(evaluation_cfg["val_interval_epochs"])
        save_interval_epochs = int(logging_cfg["save_interval_epochs"])

        train_history: list[dict] = []
        val_history: list[dict] = []

        if is_rank0(rank):
            print("starting yolov0 DDP training", flush=True)
            print("run id:", run_info["run_id"], flush=True)
            print("device:", device, flush=True)
            print("backend:", backend, flush=True)
            print("world size:", world_size, flush=True)
            print("train dataset length:", len(train_dataset), flush=True)
            print("val dataset length:", len(val_dataset), flush=True)
            print("model output shape:", output_shape, flush=True)
            print("train augmentation:", augmentation_cfg if bool(augmentation_cfg.get("enabled", False)) else "disabled", flush=True)
            print("parameter total:", param_stats["total"], flush=True)
            print("parameter trainable:", param_stats["trainable"], flush=True)
            print("tensorboard dir:", run_info["tensorboard_dir"], flush=True)
            print("output dir:", run_info["output_dir"], flush=True)
            if resume_checkpoint is not None:
                print("resume checkpoint:", resume_checkpoint, flush=True)
                print("resume start epoch:", start_epoch, flush=True)
                print("resume best epoch:", best_epoch, flush=True)
                print("resume best val loss:", best_val_loss, flush=True)
            update_metadata(
                run_info["metadata_path"],
                resume_checkpoint=str(resume_checkpoint) if resume_checkpoint is not None else "",
                resume_epoch=int(args.resume_epoch),
                resume_best_epoch=best_epoch if resume_checkpoint is not None else 0,
                resume_best_val_loss=None if best_val_loss == float("inf") else best_val_loss,
                per_rank_batch_size=per_rank_batch_size,
                effective_global_batch_size=per_rank_batch_size * world_size,
                per_rank_num_workers=per_rank_workers,
                train_augmentation=augmentation_cfg if bool(augmentation_cfg.get("enabled", False)) else {},
            )

        status = "completed"
        try:
            for epoch_index in range(start_epoch, epochs + 1):
                train_sampler.set_epoch(epoch_index)
                train_metrics = train_one_epoch_ddp(
                    model=model,
                    criterion=criterion,
                    optimizer=optimizer,
                    loader=train_loader,
                    device=device,
                    epoch_index=epoch_index,
                    writer=writer,
                    global_step=global_step,
                    log_every_steps=log_every_steps,
                    rank=rank,
                    max_steps_per_epoch=max_steps_per_epoch,
                )
                global_step = train_metrics["global_step"]
                train_history.append({"epoch": epoch_index, **train_metrics})

                if is_rank0(rank):
                    print(
                        f"[epoch {epoch_index:03d} train] "
                        f"total = {train_metrics['total_loss']:.4f} | "
                        f"box = {train_metrics['box_loss']:.4f} | "
                        f"obj = {train_metrics['obj_loss']:.4f} | "
                        f"cls = {train_metrics['cls_loss']:.4f} | "
                        f"steps = {train_metrics['optimizer_steps']} | "
                        f"global_batches = {train_metrics['global_batch_count']} | "
                        f"samples = {train_metrics['sample_count']} | "
                        f"time = {train_metrics['duration_seconds']:.1f}s",
                        flush=True,
                    )

                did_validate = False
                val_metrics = None
                if len(val_dataset) > 0 and (epoch_index % val_interval_epochs == 0 or epoch_index == epochs):
                    val_sampler.set_epoch(epoch_index)
                    val_metrics = validate_one_epoch_ddp(
                        model=model,
                        criterion=criterion,
                        loader=val_loader,
                        device=device,
                        epoch_index=epoch_index,
                        writer=writer,
                        rank=rank,
                        max_val_steps=max_val_steps,
                    )
                    val_history.append({"epoch": epoch_index, **val_metrics})
                    did_validate = True

                    if is_rank0(rank):
                        print(
                            f"[epoch {epoch_index:03d} val] "
                            f"total = {val_metrics['total_loss']:.4f} | "
                            f"box = {val_metrics['box_loss']:.4f} | "
                            f"obj = {val_metrics['obj_loss']:.4f} | "
                            f"cls = {val_metrics['cls_loss']:.4f} | "
                            f"steps = {val_metrics['optimizer_steps']} | "
                            f"global_batches = {val_metrics['global_batch_count']} | "
                            f"samples = {val_metrics['sample_count']} | "
                            f"time = {val_metrics['duration_seconds']:.1f}s",
                            flush=True,
                        )

                if is_rank0(rank):
                    if val_metrics is not None and val_metrics["total_loss"] < best_val_loss:
                        best_val_loss = val_metrics["total_loss"]
                        best_epoch = epoch_index
                        save_checkpoint_ddp(model, run_info["output_dir"] / "best.pth")

                    save_checkpoint_ddp(model, run_info["output_dir"] / "last.pth")
                    if epoch_index % save_interval_epochs == 0 or epoch_index == epochs:
                        save_checkpoint_ddp(model, run_info["output_dir"] / f"epoch_{epoch_index:03d}.pth")

                    if did_validate:
                        update_metadata(
                            run_info["metadata_path"],
                            last_epoch=epoch_index,
                            best_epoch=best_epoch,
                            best_val_loss=best_val_loss,
                            global_step=global_step,
                        )

                barrier()
                if scheduler is not None:
                    scheduler.step()
        except KeyboardInterrupt:
            status = "interrupted"
            if is_rank0(rank):
                print("training interrupted by user", flush=True)
        finally:
            if writer is not None:
                writer.close()

        if is_rank0(rank):
            visualization_dir = run_info["output_dir"] / "visualizations"
            checkpoint_for_vis = run_info["output_dir"] / "best.pth"
            if not checkpoint_for_vis.exists():
                checkpoint_for_vis = run_info["output_dir"] / "last.pth"
            if checkpoint_for_vis.exists() and len(val_dataset) > 0:
                state_dict = torch.load(checkpoint_for_vis, map_location=device)
                load_state_dict_ddp(model, state_dict)
                save_visualization_set(
                    model=base_model,
                    dataset=val_dataset,
                    output_dir=visualization_dir,
                    device=device,
                    num_classes=int(data_cfg["num_classes"]),
                    num_boxes=num_boxes,
                    anchors=anchors_by_level if anchors_by_level else anchors,
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
                f"backend = {backend}",
                f"world_size = {world_size}",
                f"device = {device}",
                f"per_rank_batch_size = {per_rank_batch_size}",
                f"effective_global_batch_size = {per_rank_batch_size * world_size}",
                f"per_rank_num_workers = {per_rank_workers}",
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
                summary_lines.append(format_history_line(item))

            summary_lines.extend(["", "val history:"])
            if val_history:
                for item in val_history:
                    summary_lines.append(format_history_line(item))
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
            print("training status:", status, flush=True)
            print("result summary:", run_info["result_path"], flush=True)

        barrier()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
