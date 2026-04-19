# Stage C Round-2 Change Notes

本文件记录 Stage C 第二轮改进的具体改动。

## 改动目标

在保留 Round-1 两项修复的前提下，进一步解决三框方案的核心分配问题：

1. 让同一个 cell 内的多个 GT 不再只抢一个 `best_anchor`
2. 建立更明确的 `正样本 / 忽略 / 负样本` 三态规则
3. 记录更多分配质量统计，方便后续分析

## 改动内容

### 1. 多 anchor 全局分配

位置：
- `data/target_encoder.py`

改动：
- 先按 cell 聚合 GT
- 先保证每个 GT 尽量拿到一个 anchor
- 再把剩余空闲 anchor 分给 IoU 达到阈值的 GT，允许额外正样本

配置开关：
- `model.anchor_positive_iou = 0.25`

### 2. Pos / Ignore / Neg 三态掩码

位置：
- `data/target_encoder.py`

改动：
- 已分配 anchor -> 正样本
- 未分配但与任意 GT 的 anchor IoU 足够高 -> Ignore
- 其他 anchor -> 负样本

### 3. 新增分配统计

位置：
- `data/target_encoder.py`
- `data/detection_dataset.py`
- `losses/yolo_loss.py`
- `engine/trainer.py`

改动：
- 新增 `ignored_count`
- 新增 `dropped_gt_count`
- TensorBoard 和结果文本同步记录

## 当前保持不变的内容

- backbone: `deep_residual`
- input size: `320 x 320`
- grid size: `10 x 10`
- num_boxes: `3`
- anchors: 保持 Round-1 不变
- box parameterization: `yolov5`
- soft objectness target: `iou`
- prediction/NMS: 本轮暂不修改

## 本轮实验目的

回答下面这个问题：

> 在保留 Round-1 修复的前提下，进一步改进 anchor 分配和 ignore 规则，是否能让三框方案真正逼近或超过单框残差基线。
