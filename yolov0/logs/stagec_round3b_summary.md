# Stage C Round-3A / Round-3B Summary

本文件总结 Stage C 第三轮改进中的前两步：

- Round-3A: Anchor 重估复核
- Round-3B: Soft objectness 冷启动保底

## 1. Round-3A：Anchor 重估复核

Round-3A 对全训练集 GT 框做了重新聚类，目的是确认当前 3 个 anchor 是否已经偏离数据分布。

对应说明文件：

- [stagec_round3a_anchor_review.md](/home/lidz/YOLO/yolov0/logs/stagec_round3a_anchor_review.md:1)

### 结论

重估得到的 3 个 anchor 与当前配置中的 anchor **完全一致**：

- `0.052141,0.076305`
- `0.199926,0.288289`
- `0.609263,0.649287`

因此：

- 当前 Stage C 的问题不是“anchor 数值完全错了”
- Round-3A 不引入新的训练 run
- 下一步直接进入 Round-3B

## 2. Round-3B：Soft Objectness 冷启动保底

### 实验目的

在保留下面这些变量不变的前提下，只修改正样本 soft objectness target 的下限：

- backbone: `deep_residual`
- input size: `320 x 320`
- grid size: `10 x 10`
- num_boxes: `3`
- anchors: 保持现有 anchor 不变
- box parameterization: `yolov5`
- anchor assignment / ignore: 保持 Round-2 版本

对应改动说明：

- [stagec_round3b_change_notes.md](/home/lidz/YOLO/yolov0/logs/stagec_round3b_change_notes.md:1)

### 对应实验

- 实验名:
  - `deep_residual_three_box_v5box_softobj_clamp`
- run_id:
  - `deep_residual_three_box_v5box_softobj_clamp_20260419_171735`
- 配置:
  - [deep_residual_three_box_v5box_softobj_clamp.toml](/home/lidz/YOLO/yolov0/configs/deep_residual_three_box_v5box_softobj_clamp.toml:1)
- 结果:
  - [result.txt](/home/lidz/YOLO/yolov0/logs/records/deep_residual_three_box_v5box_softobj_clamp_20260419_171735/result.txt:1)
- 元数据:
  - [metadata.json](/home/lidz/YOLO/yolov0/logs/records/deep_residual_three_box_v5box_softobj_clamp_20260419_171735/metadata.json:1)
- 评估:
  - [internal eval](/home/lidz/YOLO/yolov0/outputs/evaluations/deep_residual_three_box_v5box_softobj_clamp_eval.json:1)
  - [COCO subset eval](/home/lidz/YOLO/yolov0/outputs/evaluations/deep_residual_three_box_v5box_softobj_clamp_coco_eval.json:1)

## 3. 训练结果

### 核心训练统计

- 输出形状:
  - `(1, 10, 10, 255)`
- 参数量:
  - `26.56M`
- 最优 epoch:
  - `16`
- 最优验证总损失:
  - `2.673807`
- 最终训练总损失:
  - `2.439022`
- 最终验证总损失:
  - `2.723669`

### 训练观测

这轮实验最直观的变化是：

- `mean_obj_target` 整体明显抬高
- 训练前中期的验证损失下降更平稳
- 到 `epoch 8 ~ 16` 之间持续优于 Round-2 的同阶段走势

## 4. 与前两轮三框版本对比

| 指标 | Round-1 | Round-2 | Round-3B | 结论 |
| --- | ---: | ---: | ---: | --- |
| Best Val Total | `2.530998` | `2.640918` | `2.673807` | 最优仍是 Round-1 |
| Final Train Total | `2.123002` | `2.376080` | `2.439022` | 训练目标更保守 |
| Final Val Total | `2.586745` | `2.687649` | `2.723669` | 未优于 Round-1 |
| Internal mAP@0.5 | `0.021857` | `0.024256` | `0.020722` | 低于 Round-2 |
| Internal Precision | `0.026856` | `0.036024` | `0.028406` | 低于 Round-2 |
| Internal Recall | `0.199989` | `0.200331` | `0.200198` | 基本持平 |
| COCO AP50 | `0.000299` | `0.000286` | `0.000296` | 接近 Round-1，高于 Round-2 |
| COCO AR@100 | `0.001134` | `0.001014` | `0.001090` | 介于 Round-1 和 Round-2 之间 |

## 5. 与当前单框最强基线对比

当前单框最强基线仍然是：

- `deep_residual_single_box_full_loss`

对比如下：

| 指标 | Stage B 单框残差 | Stage C Round-3B 三框 | 结论 |
| --- | ---: | ---: | --- |
| Best Val Total | `2.351891` | `2.673807` | 三框仍更高 |
| Internal mAP@0.5 | `0.028802` | `0.020722` | 三框仍更低 |
| Internal Precision | `0.046880` | `0.028406` | 三框仍更低 |
| Internal Recall | `0.201682` | `0.200198` | 二者接近 |
| COCO AP50 | `0.000588` | `0.000296` | 三框仍明显更低 |

## 6. 主要结论

### 6.1 Round-3A 说明 anchor 数值本身不是当前主矛盾

Anchor 重估没有产生新 anchor，说明当前 3 个 anchor 本身就是训练集驱动的结果。
因此，后续改进重点不应放在“再盲目改 anchor 数值”，而应继续深入：

- 匹配标准
- 分配规则
- objectness / box / cls 的协同

### 6.2 Round-3B 对训练稳定性有帮助，但没有带来决定性检测收益

Objectness 下限对训练前中期收敛是正向的，验证损失曲线也更平滑。
但从正式检测指标看：

- 内部 `mAP@0.5` 低于 Round-2
- COCO `AP50` 与 Round-1 非常接近，但没有超过

因此这轮更像是一种：

- 稳定性改善
- 冷启动修复

而不是突破性性能提升。

### 6.3 当前 Stage C 的核心矛盾仍然在匹配标准本身

Round-3A 和 Round-3B 之后，最值得继续推进的下一步已经很明确：

- 不再继续微调 objectness 下限
- 不再继续重复 anchor 数值试验
- 直接进入：
  - **Shape-Matching 替代 IoU-Matching**

## 7. 当前状态

截至 Round-3B：

- Stage C 三框线已经完成多轮稳定改造与 full-run 评估
- 但当前最强实用系统仍然是：
  - `deep_residual + single-box full loss`

下一步应当进入：

- Stage C Round-3C
  - 用宽高比例匹配替代当前以 IoU 为核心的正样本判定逻辑
