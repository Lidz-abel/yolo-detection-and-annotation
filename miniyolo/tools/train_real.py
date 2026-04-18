import argparse
import sys
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


def parse_args():
    parser = argparse.ArgumentParser(description="Train MiniYOLO on real unified manifest data.")
    parser.add_argument(
        "--manifest",
        type=str,
        default="/home/lidz/YOLO/DataSet/Unified/manifests/all_val.jsonl",
        help="Path to unified manifest jsonl.",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--grid-size", type=int, default=7)
    parser.add_argument("--num-classes", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log-tag", type=str, default="real_train")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = DetectionDataset(
        manifest_path=args.manifest,
        image_size=args.image_size,
        grid_size=args.grid_size,
        num_classes=args.num_classes,
        max_samples=args.max_samples,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=detection_collate_fn,
    )

    model = MiniYOLO(num_classes=args.num_classes).to(device)
    criterion = MinimalYOLOLoss(num_classes=args.num_classes)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    log_dir = PROJECT_ROOT / "runs" / args.log_tag
    writer = SummaryWriter(log_dir=str(log_dir))

    print("device:", device)
    print("dataset length:", len(dataset))
    print("batch size:", args.batch_size)
    print("epochs:", args.epochs)
    print("log dir:", log_dir)

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_total_loss = 0.0
        epoch_cls_loss = 0.0
        epoch_box_loss = 0.0
        batch_count = 0

        for images, targets in loader:
            images = images.to(device)
            target_cls = targets["target_cls"].to(device)
            target_box = targets["target_box"].to(device)
            object_mask = targets["object_mask"].to(device)

            optimizer.zero_grad()

            pred = model(images)
            total_loss, cls_loss, box_loss = criterion(
                pred,
                target_cls,
                target_box,
                object_mask,
            )

            total_loss.backward()
            optimizer.step()

            global_step += 1
            batch_count += 1
            epoch_total_loss += total_loss.item()
            epoch_cls_loss += cls_loss.item()
            epoch_box_loss += box_loss.item()

            writer.add_scalar("loss/total_step", total_loss.item(), global_step)
            writer.add_scalar("loss/classification_step", cls_loss.item(), global_step)
            writer.add_scalar("loss/box_step", box_loss.item(), global_step)

            print(
                f"epoch {epoch:02d} | step {global_step:03d} | "
                f"total = {total_loss.item():.4f} | "
                f"cls = {cls_loss.item():.4f} | "
                f"box = {box_loss.item():.4f}"
            )

        epoch_total_loss /= max(batch_count, 1)
        epoch_cls_loss /= max(batch_count, 1)
        epoch_box_loss /= max(batch_count, 1)

        writer.add_scalar("loss/total_epoch", epoch_total_loss, epoch)
        writer.add_scalar("loss/classification_epoch", epoch_cls_loss, epoch)
        writer.add_scalar("loss/box_epoch", epoch_box_loss, epoch)

        print(
            f"[epoch {epoch:02d} summary] "
            f"total = {epoch_total_loss:.4f} | "
            f"cls = {epoch_cls_loss:.4f} | "
            f"box = {epoch_box_loss:.4f}"
        )

    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "miniyolo_real_last.pth"
    torch.save(model.state_dict(), save_path)
    print("saved model to:", save_path)

    writer.close()


if __name__ == "__main__":
    main()
