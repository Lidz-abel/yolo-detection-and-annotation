import torch
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
from models.minimal_backbone import MinimalBackbone
from models.detection_head import DetectionHead
# 用随机输入检查模型前向是否正常，并打印输入输出张量形状。
def main():
  """
  model = MinimalBackbone()
  x=torch.randn(2,3,224,224)
  y = model(x)
  print("input shape:",x.shape)
  for i, layer in enumerate(model.features):
    x=layer(x)
    print(f"layer{i}: {layer.__class__.__name__}->{x.shape}")
  print("output shape:", y.shape)
  """
  backbone=MinimalBackbone()
  head=DetectionHead(in_channels=512,num_classes=20)
  x = torch.randn(2,3,224,224)

  feature = backbone(x)
  pred = head(feature)

  class_pred = pred[..., :20]
  bbox_pred = pred[...,20:]
  print("input shape:", x.shape)
  print("feature shape:", feature.shape)
  print("prediction shape:", pred.shape)
  print("class prediction shape:", class_pred.shape)
  print("bbox prediction shape:", bbox_pred.shape)
if __name__=="__main__":
  main()


"""
input shape: torch.Size([2, 3, 224, 224])
layer0: ConvBlock->torch.Size([2, 16, 224, 224])
layer1: ConvBlock->torch.Size([2, 16, 224, 224])
layer2: MaxPool2d->torch.Size([2, 16, 112, 112])
layer3: ConvBlock->torch.Size([2, 32, 112, 112])
layer4: ConvBlock->torch.Size([2, 32, 112, 112])
layer5: MaxPool2d->torch.Size([2, 32, 56, 56])
layer6: ConvBlock->torch.Size([2, 64, 56, 56])
layer7: ConvBlock->torch.Size([2, 64, 56, 56])
layer8: MaxPool2d->torch.Size([2, 64, 28, 28])
layer9: ConvBlock->torch.Size([2, 128, 28, 28])
layer10: ConvBlock->torch.Size([2, 128, 28, 28])
layer11: MaxPool2d->torch.Size([2, 128, 14, 14])
layer12: ConvBlock->torch.Size([2, 256, 14, 14])
layer13: ConvBlock->torch.Size([2, 256, 14, 14])
layer14: MaxPool2d->torch.Size([2, 256, 7, 7])
layer15: ConvBlock->torch.Size([2, 512, 7, 7])
layer16: ConvBlock->torch.Size([2, 512, 7, 7])
output shape: torch.Size([2, 512, 7, 7])
"""
