import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from data.target_encoder import encode_target

def main():
  boxes = torch.tensor([
    [50.0,30.0,120.0,180.0]
  ])
  labels =torch.tensor([14])
  target_cls, target_box, object_mask = encode_target(
    boxes=boxes,
    labels=labels,
    image_size=224,
    grid_size=7,
    num_classes=20,
  )
  print("object_mask shape:",object_mask.shape)
  print("target_cls shape:", target_cls.shape)
  print("target_box shape:", target_box.shape)

  active_indices = torch.nonzero(object_mask)
  print("active grid indices:", active_indices)

  for idx in active_indices:
    gy,gx = idx.tolist()
    print("active cell:", gy,gx)
    print("target box:", target_box[gy,gx])
    print("activate class index:", torch.argmax(target_cls[gy,gx]).item())


if __name__=="__main__":
  main()