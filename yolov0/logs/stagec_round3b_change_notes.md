# Stage C Round-3B Change Notes

本文件记录 Stage C Round-3B 的改动内容。

## 改动目标

在保留 Round-2 三框结构、Round-1 box 参数化和 soft objectness 机制的前提下，
解决训练初期 soft objectness target 过低、导致正样本也表现得像背景的问题。

## 改动内容

### 1. 新增 soft objectness 下限

位置：

- [yolo_loss.py](/home/lidz/YOLO/yolov0/losses/yolo_loss.py:1)

改动：

- 为 `soft_objectness_target = "iou"` 的情况增加：
  - `soft_objectness_min`

当前 Round-3B 配置使用：

- `soft_objectness_min = 0.4`

具体逻辑：

- 正样本 objectness target 不再直接使用 `IoU`
- 而是使用：
  - `clamp(IoU, min=0.4, max=1.0)`

这样可以避免训练初期预测框很差时，正样本 objectness target 过低，
从而让 obj 分支过度保守。

### 2. 将该参数纳入正式配置与结果记录

位置：

- [train.py](/home/lidz/YOLO/yolov0/tools/train.py:1)
- [config.py](/home/lidz/YOLO/yolov0/utils/config.py:1)

改动：

- `soft_objectness_min` 现在由配置文件传入 loss
- `result.txt` 会记录该值，保证实验可复现

## 当前保持不变的内容

- backbone: `deep_residual`
- input size: `320 x 320`
- grid size: `10 x 10`
- num_boxes: `3`
- anchors: 保持 Round-2 不变
- box parameterization: `yolov5`
- anchor assignment / ignore 规则: 保持 Round-2 不变
- prediction/NMS: 本轮暂不修改

## 本轮实验目的

回答下面这个问题：

> 在当前三框机制下，仅仅提高 soft objectness target 的训练初期下限，是否能改善三框模型的 recall、AP50 和整体检测质量。
