# Stage C Round-1 Summary

本文件总结 Stage C 第一轮改进实验的正式结果。

## 实验目的

在保持下面这些变量不变的前提下，只修改三框方案最可能的两个核心瓶颈：

- backbone: `deep_residual`
- 输入尺寸: `320 x 320`
- 输出网格: `10 x 10`
- anchor 数量: `3`
- anchor 数值: 保持旧 Stage C 不变
- target 分配与 ignore 规则: 本轮不修改
- prediction/NMS: 本轮不修改

本轮只改：

1. `YOLOv5` 风格的 box 参数化  
2. 基于训练时 IoU 的 soft objectness target

## 对应实验

- 实验名: `deep_residual_three_box_v5box_softobj`
- run_id:
  - `deep_residual_three_box_v5box_softobj_20260419_135531`
- 配置:
  - [deep_residual_three_box_v5box_softobj.toml](/home/lidz/YOLO/yolov0/configs/deep_residual_three_box_v5box_softobj.toml:1)
- 结果:
  - [result.txt](/home/lidz/YOLO/yolov0/logs/records/deep_residual_three_box_v5box_softobj_20260419_135531/result.txt:1)
- 元数据:
  - [metadata.json](/home/lidz/YOLO/yolov0/logs/records/deep_residual_three_box_v5box_softobj_20260419_135531/metadata.json:1)
- 评估:
  - [internal eval](/home/lidz/YOLO/yolov0/outputs/evaluations/deep_residual_three_box_v5box_softobj_eval.json:1)
  - [COCO subset eval](/home/lidz/YOLO/yolov0/outputs/evaluations/deep_residual_three_box_v5box_softobj_coco_eval.json:1)

## 训练结果

### 核心训练统计

- 输出形状:
  - `(1, 10, 10, 255)`
- 参数量:
  - `26.56M`
- 最优 epoch:
  - `15`
- 最优验证总损失:
  - `2.530998`
- 最终训练总损失:
  - `2.123002`
- 最终验证总损失:
  - `2.586745`

### 验证损失走势

从 `epoch 1` 到 `epoch 15`，验证总损失从：

- `3.467373`

下降到：

- `2.530998`

后半程开始进入平台区，`epoch 15` 之后不再刷新最佳值。

## 与旧 Stage C 的直接对比

旧 Stage C:

- run_id:
  - `deep_residual_three_box_full_loss_20260419_043041`

新旧对比如下：

| 指标 | 旧 Stage C | Stage C Round-1 | 变化 |
| --- | ---: | ---: | ---: |
| Best Val Total | `2.956484` | `2.530998` | `-14.39%` |
| Final Train Total | `2.566497` | `2.123002` | `-17.28%` |
| Final Val Total | `3.202718` | `2.586745` | `-19.23%` |
| Internal mAP@0.5 | `0.016119` | `0.021857` | `+35.60%` |
| Internal Precision | `0.015502` | `0.026856` | `+73.24%` |
| Internal Recall | `0.166946` | `0.199989` | `+19.79%` |
| COCO AP50 | `0.000122` | `0.000299` | `+145.76%` |
| COCO AR@100 | `0.000999` | `0.001134` | `+13.55%` |

## 与当前最强单框基线的对比

当前单框基线:

- 实验名:
  - `deep_residual_single_box_full_loss`

对比如下：

| 指标 | Stage B 单框残差 | Stage C Round-1 三框 | 结论 |
| --- | ---: | ---: | --- |
| Best Val Total | `2.351891` | `2.530998` | 三框仍然更高 |
| Internal mAP@0.5 | `0.028802` | `0.021857` | 三框仍然更低 |
| Internal Precision | `0.046880` | `0.026856` | 三框误检更多 |
| Internal Recall | `0.201682` | `0.199989` | 二者接近 |
| COCO AP50 | `0.000588` | `0.000299` | 三框仍未超过 |

## 主要结论

### 1. 第一轮改动是有效的

当前可以明确说：

> 只修改 box 参数化和 soft objectness，就已经让三框版本出现了稳定且显著的检测质量提升。

这说明我们此前对 Stage C 头号瓶颈的判断是正确的。

### 2. 但三框版本还没有超过单框残差基线

虽然新三框版本比旧三框版强了很多，但与当前最强的：

- `deep_residual + single-box full loss`

相比，仍然存在明显差距，尤其体现在：

- `mAP@0.5`
- `Precision`
- `COCO AP50`

### 3. 下一步最值得继续改进的是 target 分配逻辑

因为本轮没有修改：

- `best-anchor only`
- `ignore` 规则
- `Pos / Neg / Ignore` 三态分配

所以当前结果很清楚地表明：

> Stage C 的“框表达能力”和“objectness 质量感知”已经明显改善，下一步最值得进入的核心战场就是 anchor 分配与 ignore 机制。

## 当前状态

本轮 Stage C Round-1 已完成并可进入正式结论。

当前可记录的结论是：

- 新三框方案已经明显优于旧三框方案
- 但仍未超过单框残差基线
- 下一步应继续深改 `encode_target()`，而不是回头再改 backbone
