# Stage C Round-1 Change Notes

本文件记录 Stage C 第一轮改进的具体改动。

## 改动目标

在不改 backbone、不改 NMS、不改 anchor 数量的前提下，先修复当前三框方案最可能的两处核心瓶颈：

1. 解除边界框宽高被 anchor 封顶的问题。  
2. 让 objectness 反映预测框质量，而不再只是硬标签 `1/0`。

## 改动内容

### 1. Box 参数化升级

位置：
- `utils/box_ops.py`

改动：
- 将三框 anchor 路径下的 box 解码从保守的 `sigmoid * anchor` 升级为更接近 YOLOv5 风格的参数化。
- 中心点改为允许跨出当前 cell 边界。
- 宽高改为允许放大到 anchor 本身尺度以上。

配置开关：
- `model.box_parameterization = "yolov5"`

### 2. Soft Objectness Target

位置：
- `losses/yolo_loss.py`

改动：
- 正样本 objectness 不再固定为 `1`
- 改为使用训练时预测框与 GT 框的实时 IoU 作为 soft target
- 背景仍保持 `0`

配置开关：
- `loss.soft_objectness_target = "iou"`

## 当前保持不变的内容

- backbone: `deep_residual`
- input size: `320 x 320`
- grid size: `10 x 10`
- num_boxes: `3`
- anchors: 保持上一版 Stage C 不变
- target 分配与 ignore 规则：本轮暂不修改
- prediction/NMS：本轮暂不修改

## 本轮实验目的

回答下面这个问题：

> 在不动分配规则的情况下，只修复 box 表达能力和 objectness 质量感知，是否足以显著改善三框版本的检测效果。
