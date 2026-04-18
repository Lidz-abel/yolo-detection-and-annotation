import torch
import torch.nn as nn

class DetectionHead(nn.Module):
  def __init__(self,in_channels=512,num_classes=20):
    super().__init__()
    self.num_classes = num_classes
    self.pred_dim = num_classes+4
    self.head = nn.Sequential(
      nn.Conv2d(in_channels, 256, kernel_size=3,stride=1,padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(256,self.pred_dim,kernel_size=1,stride=1)
    )
  def forward(self,x):
    x=self.head(x)
    x=x.permute(0,2,3,1).contiguous()
    return x