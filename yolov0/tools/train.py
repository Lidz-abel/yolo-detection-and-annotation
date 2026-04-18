"""Baseline training entrypoint for yolov0 stage-two development."""

import argparse
import random
import sys
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from data.detection_dataset import DetectionDataset, detection_collate_fn
from losses.detection_loss import DetectionLoss
from models.detector import YOLOv0Baseline
from utils.config import load_config, summarize_config
from utils.experiment import init_run, update_metadata, write_result_summary
from utils.modeling import count_parameters, describe_model_output


def parse_args():
    """Parse the config path and optional runtime seed override."""
    parser = argparse.ArgumentParser(description="Baseline training entry for yolov0.")
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
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Seed Python and torch RNGs for reproducible baseline experiments."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_device(device_name: str):
    """Pick the requested device, or auto-select CUDA when available."""
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def main():
    """Bootstrap or smoke-train the first yolov0 baseline from config."""
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)

    data_cfg = config["data"]
    model_cfg = config["model"]
    loss_cfg = config["loss"]
    train_cfg = config["train"]

    if args.seed is not None:
        train_cfg["seed"] = args.seed
    set_seed(int(train_cfg["seed"]))
    device = build_device(str(train_cfg["device"]))

    run_info = init_run(PROJECT_ROOT, config_path, config)
    update_metadata(
        run_info["metadata_path"],
        status="baseline_ready",
        stage="baseline_training",
        device=str(device),
    )

    dataset = DetectionDataset(
        manifest_path=data_cfg["train_manifest"],
        image_size=int(data_cfg["image_size"]),
        grid_size=int(data_cfg["grid_size"]),
        num_classes=int(data_cfg["num_classes"]),
        max_samples=int(data_cfg["train_max_samples"]),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=bool(train_cfg["shuffle"]),
        num_workers=int(train_cfg["num_workers"]),
        pin_memory=bool(train_cfg["pin_memory"]),
        persistent_workers=bool(train_cfg["persistent_workers"]) if int(train_cfg["num_workers"]) > 0 else False,
        prefetch_factor=int(train_cfg["prefetch_factor"]) if int(train_cfg["num_workers"]) > 0 else None,
        collate_fn=detection_collate_fn,
    )

    model = YOLOv0Baseline(
        num_classes=int(data_cfg["num_classes"]),
        width_mult=float(model_cfg["width_mult"]),
        depth_mult=float(model_cfg["depth_mult"]),
        use_residual=bool(model_cfg["use_residual"]),
    ).to(device)
    criterion = DetectionLoss(
        num_classes=int(data_cfg["num_classes"]),
        lambda_cls=float(loss_cfg["lambda_cls"]),
        lambda_box=float(loss_cfg["lambda_box"]),
    )
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )

    param_stats = count_parameters(model)
    output_shape = describe_model_output(model, int(data_cfg["image_size"]), device)

    first_batch_total = None
    first_batch_cls = None
    first_batch_box = None
    first_batch_shape = None
    if len(dataset) > 0:
        images, targets = next(iter(loader))
        images = images.to(device)
        target_cls = targets["target_cls"].to(device)
        target_box = targets["target_box"].to(device)
        object_mask = targets["object_mask"].to(device)

        pred = model(images)
        total_loss, cls_loss, box_loss = criterion(pred, target_cls, target_box, object_mask)
        first_batch_total = float(total_loss.item())
        first_batch_cls = float(cls_loss.item())
        first_batch_box = float(box_loss.item())
        first_batch_shape = tuple(images.shape)

        # A single optimizer step proves the baseline path is already trainable.
        if not bool(train_cfg["bootstrap_only"]):
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

    summary_lines = [
        f"run_id = {run_info['run_id']}",
        f"timestamp = {run_info['timestamp']}",
        f"status = baseline_ready",
        f"stage = baseline_training",
        f"project_root = {PROJECT_ROOT}",
        f"config_path = {config_path}",
        f"config_snapshot = {run_info['config_snapshot']}",
        f"tensorboard_dir = {run_info['tensorboard_dir']}",
        f"output_dir = {run_info['output_dir']}",
        f"record_dir = {run_info['record_dir']}",
        f"git_commit = {run_info['metadata']['git_commit']}",
        f"device = {device}",
        f"dataset_length = {len(dataset)}",
        f"model_output_shape = {output_shape}",
        f"parameter_total = {param_stats['total']}",
        f"parameter_trainable = {param_stats['trainable']}",
        f"bootstrap_only = {bool(train_cfg['bootstrap_only'])}",
        "",
        "config summary:",
        *summarize_config(config),
        "",
        "smoke summary:",
        f"first_batch_shape = {first_batch_shape}",
        f"first_batch_total_loss = {first_batch_total}",
        f"first_batch_cls_loss = {first_batch_cls}",
        f"first_batch_box_loss = {first_batch_box}",
        "",
        "note:",
        "stage two baseline components are now connected: dataset, detector, loss, and optimizer.",
        "this entry currently performs a smoke pass by default and can be upgraded to full training next.",
    ]

    write_result_summary(run_info["result_path"], summary_lines)

    print("initialized yolov0 baseline")
    print("run id:", run_info["run_id"])
    print("device:", device)
    print("dataset length:", len(dataset))
    print("model output shape:", output_shape)
    print("parameter total:", param_stats["total"])
    print("parameter trainable:", param_stats["trainable"])
    print("first batch total loss:", first_batch_total)
    print("config snapshot:", run_info["config_snapshot"])
    print("metadata:", run_info["metadata_path"])
    print("result summary:", run_info["result_path"])
    print("tensorboard dir:", run_info["tensorboard_dir"])
    print("output dir:", run_info["output_dir"])


if __name__ == "__main__":
    main()
