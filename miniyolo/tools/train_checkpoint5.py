import argparse
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from data.detection_dataset import DetectionDataset, detection_collate_fn
from losses.minimal_yolo_loss import MinimalYOLOLoss
from models.miniyolo import MiniYOLO
from utils.simple_toml import load_toml


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


def main():
    args = parse_args()
    config = load_toml(args.config)

    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["train"]
    logging_cfg = config["logging"]

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
        collate_fn=detection_collate_fn,
    )

    model = MiniYOLO(num_classes=int(data_cfg["num_classes"])).to(device)
    criterion = MinimalYOLOLoss(
        num_classes=int(data_cfg["num_classes"]),
        lambda_cls=float(model_cfg["lambda_cls"]),
        lambda_box=float(model_cfg["lambda_box"]),
    )
    optimizer = optim.Adam(model.parameters(), lr=float(train_cfg["lr"]))

    runs_dir = Path(logging_cfg["runs_dir"])
    output_dir = Path(logging_cfg["output_dir"])
    results_dir = Path(logging_cfg["results_dir"])
    for path in (runs_dir, output_dir, results_dir):
        path.mkdir(parents=True, exist_ok=True)

    run_name = str(logging_cfg["run_name"])
    log_every_steps = int(logging_cfg["log_every_steps"])
    writer = SummaryWriter(log_dir=str(runs_dir / run_name))

    print("device:", device)
    print("config:", args.config)
    print("run name:", run_name)
    print("train dataset length:", len(train_dataset))
    print("batch size:", int(train_cfg["batch_size"]))
    print("epochs:", int(train_cfg["epochs"]))

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
            optimizer.step()

            lr_value = optimizer.param_groups[0]["lr"]
            global_step += 1
            batch_count += 1
            epoch_total_loss += total_loss.item()
            epoch_cls_loss += cls_loss.item()
            epoch_box_loss += box_loss.item()

            writer.add_scalar("loss/total_step", total_loss.item(), global_step)
            writer.add_scalar("loss/classification_step", cls_loss.item(), global_step)
            writer.add_scalar("loss/box_step", box_loss.item(), global_step)
            writer.add_scalar("train/lr", lr_value, global_step)

            if global_step == 1 or global_step % log_every_steps == 0:
                print(
                    f"epoch {epoch:02d} | step {global_step:03d} | "
                    f"total = {total_loss.item():.4f} | "
                    f"cls = {cls_loss.item():.4f} | "
                    f"box = {box_loss.item():.4f} | "
                    f"lr = {lr_value:.6f}"
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
            }
        )

        writer.add_scalar("loss/total_epoch", epoch_total_loss, epoch)
        writer.add_scalar("loss/classification_epoch", epoch_cls_loss, epoch)
        writer.add_scalar("loss/box_epoch", epoch_box_loss, epoch)

        print(
            f"[epoch {epoch:02d} summary] "
            f"total = {epoch_total_loss:.4f} | "
            f"cls = {epoch_cls_loss:.4f} | "
            f"box = {epoch_box_loss:.4f}"
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = output_dir / f"{run_name}_{timestamp}.pth"
    result_path = results_dir / f"{run_name}_{timestamp}.txt"
    config_snapshot_path = results_dir / f"{run_name}_{timestamp}.toml"

    torch.save(model.state_dict(), model_path)
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
        f"num_workers = {int(train_cfg['num_workers'])}",
        f"model_path = {model_path}",
        f"config_snapshot = {config_snapshot_path}",
        "",
        "epoch summaries:",
    ]
    for summary in epoch_summaries:
        lines.append(
            f"epoch {summary['epoch']:02d}: "
            f"total={summary['total_loss']:.4f}, "
            f"cls={summary['cls_loss']:.4f}, "
            f"box={summary['box_loss']:.4f}"
        )

    result_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("saved model to:", model_path)
    print("saved result txt to:", result_path)
    print("saved config snapshot to:", config_snapshot_path)
    writer.close()


if __name__ == "__main__":
    main()
