import sys
from pathlib import Path

import torch
import torch.optim as optim

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from data.target_encoder import encode_target
from losses.minimal_yolo_loss import MinimalYOLOLoss
from models.miniyolo import MiniYOLO


def build_toy_batch(batch_size=2, image_size=224, num_classes=20):
    images = torch.randn(batch_size, 3, image_size, image_size)

    batch_target_cls = []
    batch_target_box = []
    batch_object_mask = []

    toy_boxes = [
        torch.tensor([[50.0, 30.0, 120.0, 180.0]]),
        torch.tensor([[120.0, 60.0, 190.0, 200.0]]),
    ]
    toy_labels = [
        torch.tensor([14]),
        torch.tensor([7]),
    ]

    for index in range(batch_size):
        boxes = toy_boxes[index % len(toy_boxes)]
        labels = toy_labels[index % len(toy_labels)]
        target_cls, target_box, object_mask = encode_target(
            boxes=boxes,
            labels=labels,
            image_size=image_size,
            grid_size=7,
            num_classes=num_classes,
        )
        batch_target_cls.append(target_cls)
        batch_target_box.append(target_box)
        batch_object_mask.append(object_mask)

    target_cls = torch.stack(batch_target_cls, dim=0)
    target_box = torch.stack(batch_target_box, dim=0)
    object_mask = torch.stack(batch_object_mask, dim=0)
    return images, target_cls, target_box, object_mask


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MiniYOLO(num_classes=20).to(device)
    criterion = MinimalYOLOLoss(num_classes=20)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    images, target_cls, target_box, object_mask = build_toy_batch(batch_size=2)
    images = images.to(device)
    target_cls = target_cls.to(device)
    target_box = target_box.to(device)
    object_mask = object_mask.to(device)

    print("device:", device)
    print("images shape:", images.shape)
    print("target_cls shape:", target_cls.shape)
    print("target_box shape:", target_box.shape)
    print("object_mask shape:", object_mask.shape)

    num_steps = 30
    for step in range(1, num_steps + 1):
        optimizer.zero_grad()

        pred = model(images)
        loss = criterion(pred, target_cls, target_box, object_mask)

        loss.backward()
        optimizer.step()

        if step == 1 or step % 5 == 0:
            print(f"step {step:02d} | loss = {loss.item():.4f}")


if __name__ == "__main__":
    main()
