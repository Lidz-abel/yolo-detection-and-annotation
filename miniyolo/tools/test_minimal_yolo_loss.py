import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from data.target_encoder import encode_target
from losses.minimal_yolo_loss import MinimalYOLOLoss


def main():
    pred = torch.randn(2, 7, 7, 24, requires_grad=True)

    boxes = torch.tensor([
        [50.0, 30.0, 120.0, 180.0]
    ])
    labels = torch.tensor([14])

    target_cls, target_box, object_mask = encode_target(
        boxes=boxes,
        labels=labels,
        image_size=224,
        grid_size=7,
        num_classes=20,
    )

    target_cls = target_cls.unsqueeze(0).repeat(2, 1, 1, 1)
    target_box = target_box.unsqueeze(0).repeat(2, 1, 1, 1)
    object_mask = object_mask.unsqueeze(0).repeat(2, 1, 1)

    criterion = MinimalYOLOLoss(num_classes=20)
    loss = criterion(pred, target_cls, target_box, object_mask)

    print("pred shape:", pred.shape)
    print("target_cls shape:", target_cls.shape)
    print("target_box shape:", target_box.shape)
    print("object_mask shape:", object_mask.shape)
    print("loss:", loss)

    loss.backward()
    print("grad shape:", pred.grad.shape)


if __name__ == "__main__":
    main()