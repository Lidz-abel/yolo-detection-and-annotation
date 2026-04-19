# Stage C Round-2 Summary

本文件总结 Stage C 第二轮改进实验的正式结果。

## 实验目的

在保留 Round-1 两项修复的前提下：

- `YOLOv5` 风格 box 参数化
- IoU-based soft objectness target

进一步改进三框方案中的 anchor 分配与忽略规则，回答下面这个问题：

> 更合理的 `Pos / Ignore / Neg` 三态分配，是否能让三框版本进一步逼近或超过当前的单框残差基线。

## 对应实验

- 实验名:
  - `deep_residual_three_box_v5box_softobj_assign`
- run_id:
  - `deep_residual_three_box_v5box_softobj_assign_20260419_152713`
- 配置:
  - [deep_residual_three_box_v5box_softobj_assign.toml](/home/lidz/YOLO/yolov0/configs/deep_residual_three_box_v5box_softobj_assign.toml:1)
- 结果:
  - [result.txt](/home/lidz/YOLO/yolov0/logs/records/deep_residual_three_box_v5box_softobj_assign_20260419_152713/result.txt:1)
- 元数据:
  - [metadata.json](/home/lidz/YOLO/yolov0/logs/records/deep_residual_three_box_v5box_softobj_assign_20260419_152713/metadata.json:1)
- 评估:
  - [internal eval](/home/lidz/YOLO/yolov0/outputs/evaluations/deep_residual_three_box_v5box_softobj_assign_eval.json:1)
  - [COCO subset eval](/home/lidz/YOLO/yolov0/outputs/evaluations/deep_residual_three_box_v5box_softobj_assign_coco_eval.json:1)

## 本轮改动

位置主要在：

- [target_encoder.py](/home/lidz/YOLO/yolov0/data/target_encoder.py:1)
- [detection_dataset.py](/home/lidz/YOLO/yolov0/data/detection_dataset.py:1)
- [yolo_loss.py](/home/lidz/YOLO/yolov0/losses/yolo_loss.py:1)
- [trainer.py](/home/lidz/YOLO/yolov0/engine/trainer.py:1)

改动内容：

1. 按 cell 聚合 GT，不再只按单个 GT 独立寻找 `best_anchor`
2. 先保证每个 GT 尽量获得一个空闲 anchor
3. 再把剩余空闲 anchor 分给 IoU 达到阈值的 GT
4. 明确区分：
   - 正样本
   - 忽略样本
   - 负样本
5. 新增并记录：
   - `ignored_count`
   - `dropped_gt_count`
   - `collision_count`

## 训练结果

### 核心训练统计

- 输出形状:
  - `(1, 10, 10, 255)`
- 参数量:
  - `26.56M`
- 最优 epoch:
  - `16`
- 最优验证总损失:
  - `2.640918`
- 最终训练总损失:
  - `2.376080`
- 最终验证总损失:
  - `2.687649`

### 分配统计

从完整训练结果看：

- `ignored_count` 基本为 `0`
- `dropped_gt_count`
  - train 约为 `0.1587 / batch`
  - val 约为 `0.0823 / batch`
- `collision_count`
  - train 约为 `1.368 / batch`
  - val 约为 `0.757 / batch`

这说明：

- 当前新分配逻辑基本没有产生 ignore 样本
- 同一 cell 的 GT 冲突仍然持续存在
- 多框分配的主要矛盾，仍然集中在 crowded cells，而不是 ignore 规则本身

## 与 Stage C Round-1 的对比

Stage C Round-1:

- run_id:
  - `deep_residual_three_box_v5box_softobj_20260419_135531`

对比如下：

| 指标 | Round-1 | Round-2 | 变化 |
| --- | ---: | ---: | ---: |
| Best Val Total | `2.530998` | `2.640918` | `+4.34%` |
| Final Train Total | `2.123002` | `2.376080` | `+11.92%` |
| Final Val Total | `2.586745` | `2.687649` | `+3.90%` |
| Internal mAP@0.5 | `0.021857` | `0.024256` | `+10.98%` |
| Internal Precision | `0.026856` | `0.036024` | `+34.14%` |
| Internal Recall | `0.199989` | `0.200331` | `+0.17%` |
| COCO AP50 | `0.000299` | `0.000286` | `-4.30%` |
| COCO AR@100 | `0.001134` | `0.001014` | `-10.64%` |

### 解释

Round-2 的结果呈现出一个比较典型的“内部指标改善，但 COCO 子集官方风格指标未同步改善”的现象：

- 内部 `mAP@0.5`、`precision` 提升
- COCO `AP50`、`AR@100` 略降

这说明：

- 新的分配逻辑确实让预测分布发生了变化
- 但这些变化还没有稳定转化成更高质量的跨阈值检测结果

## 与当前最强单框基线的对比

当前单框基线:

- 实验名:
  - `deep_residual_single_box_full_loss`

对比如下：

| 指标 | Stage B 单框残差 | Stage C Round-2 三框 | 结论 |
| --- | ---: | ---: | --- |
| Best Val Total | `2.351891` | `2.640918` | 三框仍更高 |
| Internal mAP@0.5 | `0.028802` | `0.024256` | 三框仍更低 |
| Internal Precision | `0.046880` | `0.036024` | 三框仍更低 |
| Internal Recall | `0.201682` | `0.200331` | 二者接近 |
| COCO AP50 | `0.000588` | `0.000286` | 三框仍明显更低 |

## 主要结论

### 1. Round-2 训练稳定，但没有带来决定性收益

这轮训练是稳定的：

- 无 NaN
- loss 正常下降
- full-run 成功完成

但从正式检测结果看：

- 它没有超过 Round-1
- 也没有接近或超过当前单框残差基线

### 2. 当前 Stage C 的主要问题不再只是“有没有三态规则”

Round-2 已经说明：

- 仅仅把 `best-anchor only` 改成当前这版更宽松的分配策略
- 并不足以让三框方案真正胜出

所以当前更值得怀疑的地方是：

- anchor 尺寸本身
- `anchor_positive_iou` 与 `anchor_ignore_iou` 的设计
- objectness 与 box 质量的耦合方式
- crowded cells 下的多 GT 分配策略仍然不够强

### 3. 当前最强实用系统仍然是 Stage B

到目前为止，最强实用基线仍然是：

- `deep_residual + single-box full loss`

Stage C 这条线已经从：

- 原始三框
- Round-1
- Round-2

连续推进了两轮，但还没有实现超越。

## 当前状态

Stage C Round-2 已完成并进入正式记录。

当前可记录结论是：

- Round-2 不是无效改动，它提高了内部 `mAP@0.5` 和 `precision`
- 但它没有把三框线推到超过 Round-1，更没有超过单框残差基线
- 下一步如果继续推进三框，应该优先考虑：
  - 重新审视 anchor 设计
  - 重新审视 objectness / matching 的质量耦合
  - 而不是继续只在现有三态分配逻辑上做局部修补
