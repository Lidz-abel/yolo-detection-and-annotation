import argparse
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from data.detection_dataset import DetectionDataset, detection_collate_fn
from losses.minimal_yolo_loss import MinimalYOLOLoss
from models.miniyolo import MiniYOLO
from utils.metrics import evaluate_map50
from utils.simple_toml import load_toml
from utils.visualization import save_progress_visualizations


def parse_args():
    parser = argparse.ArgumentParser(description="Checkpoint 5 training entry with TOML config.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "train_base.toml"),
        help="Path to training TOML config.",
    )
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_device(device_name: str):
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def build_scheduler(optimizer, scheduler_name: str, epochs: int, min_lr: float):
    name = scheduler_name.lower()
    if name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=min_lr)
    return None


def log_gradient_norms(model, writer, global_step):
    total_sq_norm = 0.0
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        grad_norm = param.grad.detach().norm(2).item()
        total_sq_norm += grad_norm ** 2
        writer.add_scalar(f"grad_norm/{name.replace('.', '/')}", grad_norm, global_step)
    writer.add_scalar("grad_norm/global_total", total_sq_norm ** 0.5, global_step)


def main():
    args = parse_args()
    config = load_toml(args.config)

    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["train"]
    logging_cfg = config["logging"]
    evaluation_cfg = config["evaluation"]
    visualization_cfg = config["visualization"]

    set_seed(int(train_cfg["seed"]))
    device = build_device(str(train_cfg["device"]))

    train_dataset = DetectionDataset(
        manifest_path=data_cfg["train_manifest"],
        image_size=int(data_cfg["image_size"]),
        grid_size=int(data_cfg["grid_size"]),
        num_classes=int(data_cfg["num_classes"]),
        max_samples=int(data_cfg["train_max_samples"]),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=bool(train_cfg["shuffle"]),
        num_workers=int(train_cfg["num_workers"]),
        pin_memory=bool(train_cfg["pin_memory"]),
        persistent_workers=bool(train_cfg["persistent_workers"]) if int(train_cfg["num_workers"]) > 0 else False,
        prefetch_factor=int(train_cfg["prefetch_factor"]) if int(train_cfg["num_workers"]) > 0 else None,
        collate_fn=detection_collate_fn,
    )

    val_dataset = DetectionDataset(
        manifest_path=data_cfg["val_manifest"],
        image_size=int(data_cfg["image_size"]),
        grid_size=int(data_cfg["grid_size"]),
        num_classes=int(data_cfg["num_classes"]),
        max_samples=int(data_cfg["val_max_samples"]),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(train_cfg["num_workers"]),
        pin_memory=bool(train_cfg["pin_memory"]),
        persistent_workers=bool(train_cfg["persistent_workers"]) if int(train_cfg["num_workers"]) > 0 else False,
        prefetch_factor=int(train_cfg["prefetch_factor"]) if int(train_cfg["num_workers"]) > 0 else None,
        collate_fn=detection_collate_fn,
    )

    model = MiniYOLO(num_classes=int(data_cfg["num_classes"])).to(device)
    use_data_parallel = (
        bool(train_cfg["use_data_parallel"])
        and torch.cuda.is_available()
        and torch.cuda.device_count() > 1
    )
    if use_data_parallel:
        model = nn.DataParallel(model)
    criterion = MinimalYOLOLoss(
        num_classes=int(data_cfg["num_classes"]),
        lambda_cls=float(model_cfg["lambda_cls"]),
        lambda_box=float(model_cfg["lambda_box"]),
    )
    optimizer = optim.Adam(model.parameters(), lr=float(train_cfg["lr"]))
    scheduler = build_scheduler(
        optimizer=optimizer,
        scheduler_name=str(train_cfg["scheduler"]),
        epochs=int(train_cfg["epochs"]),
        min_lr=float(train_cfg["min_lr"]),
    )

    runs_dir = Path(logging_cfg["runs_dir"])
    output_dir = Path(logging_cfg["output_dir"])
    results_dir = Path(logging_cfg["results_dir"])
    for path in (runs_dir, output_dir, results_dir):
        path.mkdir(parents=True, exist_ok=True)

    run_name = str(logging_cfg["run_name"])
    log_every_steps = int(logging_cfg["log_every_steps"])
    grad_log_interval_steps = int(logging_cfg["grad_log_interval_steps"])
    writer = SummaryWriter(log_dir=str(runs_dir / run_name))
    progress_output_dir = output_dir / f"{run_name}_progress"
    progress_output_dir.mkdir(parents=True, exist_ok=True)

    print("device:", device)
    print("config:", args.config)
    print("run name:", run_name)
    print("train dataset length:", len(train_dataset))
    print("val dataset length:", len(val_dataset))
    print("batch size:", int(train_cfg["batch_size"]))
    print("epochs:", int(train_cfg["epochs"]))
    print("num_workers:", int(train_cfg["num_workers"]))
    print("pin_memory:", bool(train_cfg["pin_memory"]))
    print("persistent_workers:", bool(train_cfg["persistent_workers"]))
    print("prefetch_factor:", int(train_cfg["prefetch_factor"]))
    print("gpu_count:", torch.cuda.device_count() if torch.cuda.is_available() else 0)
    print("data_parallel:", use_data_parallel)

    initial_vis_dir = save_progress_visualizations(
        model=model,
        dataset=val_dataset,
        device=device,
        output_root=progress_output_dir,
        epoch=0,
        image_size=int(data_cfg["image_size"]),
        grid_size=int(data_cfg["grid_size"]),
        num_classes=int(data_cfg["num_classes"]),
        top_k=int(evaluation_cfg["eval_top_k"]),
        score_threshold=float(evaluation_cfg["eval_score_threshold"]),
        max_samples=int(visualization_cfg["max_visualization_samples"]),
    )
    print("saved initial progress visualization to:", initial_vis_dir)

    global_step = 0
    epoch_summaries = []
    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        model.train()
        epoch_total_loss = 0.0
        epoch_cls_loss = 0.0
        epoch_box_loss = 0.0
        batch_count = 0

        for images, targets in train_loader:
            images = images.to(device)
            target_cls = targets["target_cls"].to(device)
            target_box = targets["target_box"].to(device)
            object_mask = targets["object_mask"].to(device)

            optimizer.zero_grad()
            pred = model(images)
            total_loss, cls_loss, box_loss = criterion(pred, target_cls, target_box, object_mask)
            total_loss.backward()
            if global_step == 0 or (global_step + 1) % grad_log_interval_steps == 0:
                log_gradient_norms(model, writer, global_step + 1)
            optimizer.step()

            lr_value = optimizer.param_groups[0]["lr"]
            global_step += 1
            batch_count += 1
            epoch_total_loss += total_loss.item()
            epoch_cls_loss += cls_loss.item()
            epoch_box_loss += box_loss.item()
            positive_cells = object_mask.sum(dim=(1, 2)).float().mean().item()

            writer.add_scalar("loss/total_step", total_loss.item(), global_step)
            writer.add_scalar("loss/classification_step", cls_loss.item(), global_step)
            writer.add_scalar("loss/box_step", box_loss.item(), global_step)
            writer.add_scalar("train/lr", lr_value, global_step)
            writer.add_scalar("train/positive_cells_per_image", positive_cells, global_step)

            if global_step == 1 or global_step % log_every_steps == 0:
                print(
                    f"epoch {epoch:02d} | step {global_step:03d} | "
                    f"total = {total_loss.item():.4f} | "
                    f"cls = {cls_loss.item():.4f} | "
                    f"box = {box_loss.item():.4f} | "
                    f"lr = {lr_value:.6f} | "
                    f"pos_cells/img = {positive_cells:.2f}"
                )

        epoch_total_loss /= max(batch_count, 1)
        epoch_cls_loss /= max(batch_count, 1)
        epoch_box_loss /= max(batch_count, 1)
        epoch_summaries.append(
            {
                "epoch": epoch,
                "total_loss": epoch_total_loss,
                "cls_loss": epoch_cls_loss,
                "box_loss": epoch_box_loss,
                "map50": None,
                "mean_pred_score": None,
                "num_predictions": None,
            }
        )

        writer.add_scalar("loss/total_epoch", epoch_total_loss, epoch)
        writer.add_scalar("loss/classification_epoch", epoch_cls_loss, epoch)
        writer.add_scalar("loss/box_epoch", epoch_box_loss, epoch)
        writer.add_scalar("train/lr_epoch", optimizer.param_groups[0]["lr"], epoch)

        print(
            f"[epoch {epoch:02d} summary] "
            f"total = {epoch_total_loss:.4f} | "
            f"cls = {epoch_cls_loss:.4f} | "
            f"box = {epoch_box_loss:.4f}"
        )

        if epoch % int(evaluation_cfg["eval_interval_epochs"]) == 0:
            eval_metrics = evaluate_map50(
                model=model,
                loader=val_loader,
                device=device,
                image_size=int(data_cfg["image_size"]),
                grid_size=int(data_cfg["grid_size"]),
                num_classes=int(data_cfg["num_classes"]),
                iou_threshold=float(evaluation_cfg["map_iou_threshold"]),
                top_k=int(evaluation_cfg["eval_top_k"]),
                score_threshold=float(evaluation_cfg["eval_score_threshold"]),
            )
            writer.add_scalar("val/map50", eval_metrics["map50"], epoch)
            writer.add_scalar("val/mean_pred_score", eval_metrics["mean_pred_score"], epoch)
            writer.add_scalar("val/num_predictions", eval_metrics["num_predictions"], epoch)
            writer.add_scalar("val/num_present_classes", eval_metrics["num_present_classes"], epoch)
            epoch_summaries[-1]["map50"] = eval_metrics["map50"]
            epoch_summaries[-1]["mean_pred_score"] = eval_metrics["mean_pred_score"]
            epoch_summaries[-1]["num_predictions"] = eval_metrics["num_predictions"]

            print(
                f"[epoch {epoch:02d} eval] "
                f"mAP@0.5 = {eval_metrics['map50']:.4f} | "
                f"mean_pred_score = {eval_metrics['mean_pred_score']:.4f} | "
                f"predictions = {eval_metrics['num_predictions']}"
            )

        if scheduler is not None:
            scheduler.step()

        if epoch % int(visualization_cfg["vis_interval_epochs"]) == 0:
            vis_dir = save_progress_visualizations(
                model=model,
                dataset=val_dataset,
                device=device,
                output_root=progress_output_dir,
                epoch=epoch,
                image_size=int(data_cfg["image_size"]),
                grid_size=int(data_cfg["grid_size"]),
                num_classes=int(data_cfg["num_classes"]),
                top_k=int(evaluation_cfg["eval_top_k"]),
                score_threshold=float(evaluation_cfg["eval_score_threshold"]),
                max_samples=int(visualization_cfg["max_visualization_samples"]),
            )
            print(f"[epoch {epoch:02d} vis] saved progress snapshots to:", vis_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = output_dir / f"{run_name}_{timestamp}.pth"
    result_path = results_dir / f"{run_name}_{timestamp}.txt"
    config_snapshot_path = results_dir / f"{run_name}_{timestamp}.toml"

    state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    torch.save(state_dict, model_path)
    shutil.copy2(args.config, config_snapshot_path)

    lines = [
        f"run_name = {run_name}",
        f"timestamp = {timestamp}",
        f"config = {args.config}",
        f"device = {device}",
        f"train_manifest = {data_cfg['train_manifest']}",
        f"image_size = {int(data_cfg['image_size'])}",
        f"grid_size = {int(data_cfg['grid_size'])}",
        f"num_classes = {int(data_cfg['num_classes'])}",
        f"train_max_samples = {int(data_cfg['train_max_samples'])}",
        f"batch_size = {int(train_cfg['batch_size'])}",
        f"epochs = {int(train_cfg['epochs'])}",
        f"lr = {float(train_cfg['lr'])}",
        f"min_lr = {float(train_cfg['min_lr'])}",
        f"scheduler = {str(train_cfg['scheduler'])}",
        f"num_workers = {int(train_cfg['num_workers'])}",
        f"pin_memory = {bool(train_cfg['pin_memory'])}",
        f"persistent_workers = {bool(train_cfg['persistent_workers'])}",
        f"prefetch_factor = {int(train_cfg['prefetch_factor'])}",
        f"use_data_parallel = {bool(train_cfg['use_data_parallel'])}",
        f"eval_interval_epochs = {int(evaluation_cfg['eval_interval_epochs'])}",
        f"map_iou_threshold = {float(evaluation_cfg['map_iou_threshold'])}",
        f"model_path = {model_path}",
        f"config_snapshot = {config_snapshot_path}",
        f"progress_output_dir = {progress_output_dir}",
        "",
        "epoch summaries:",
    ]
    for summary in epoch_summaries:
        line = (
            f"epoch {summary['epoch']:02d}: "
            f"total={summary['total_loss']:.4f}, "
            f"cls={summary['cls_loss']:.4f}, "
            f"box={summary['box_loss']:.4f}"
        )
        if summary["map50"] is not None:
            line += (
                f", map50={summary['map50']:.4f}, "
                f"mean_pred_score={summary['mean_pred_score']:.4f}, "
                f"num_predictions={summary['num_predictions']}"
            )
        lines.append(line)

    result_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("saved model to:", model_path)
    print("saved result txt to:", result_path)
    print("saved config snapshot to:", config_snapshot_path)
    writer.close()


if __name__ == "__main__":
    main()
