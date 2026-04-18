import torch
import torch.nn as nn

# 基本卷积块：一层卷积后接一层 ReLU 激活。
class ConvBlock(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.conv=nn.Conv2d(in_channels, out_channels, kernel_size=3, stride =1 , padding =1)
    self.relu=nn.ReLU(inplace = True)
  
  def forward(self,x):
    x=self.conv(x)
    x=self.relu(x)
    return x

# 极简 backbone：通过多层卷积和池化把输入图像压缩到更小的特征图。
class MinimalBackbone(nn.Module):
  def __init__(self):
    super().__init__()

    self.features = nn.Sequential(
      ConvBlock(3,16),
      ConvBlock(16,16),
      nn.MaxPool2d(kernel_size=2, stride=2),

      ConvBlock(16,32),
      ConvBlock(32,32),
      nn.MaxPool2d(kernel_size=2, stride=2),

      ConvBlock(32,64),
      ConvBlock(64,64),
      nn.MaxPool2d(kernel_size=2, stride=2),

      ConvBlock(64,128),
      ConvBlock(128,128),
      nn.MaxPool2d(kernel_size=2, stride =2),

      ConvBlock(128, 256),
      ConvBlock(256, 256),
      nn.MaxPool2d(kernel_size=2, stride=2),
      
      ConvBlock(256,512),
      ConvBlock(512,512),
    )
  def forward(self,x):
    return self.features(x)
