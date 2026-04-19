# Stage C Round-3C Summary

本文件总结 Stage C 第三轮改进中的最后一步：

- `Shape-Matching` 替代 `IoU-Matching`

## 1. 实验目的

Round-3C 的目标是继续沿着最小主变量推进三框机制：

- backbone 保持 `deep_residual`
- 输入尺寸保持 `320 x 320`
- 输出网格保持 `10 x 10`
- 三框 head 保持 `num_boxes = 3`
- anchors 保持 Round-3A 复核后的现有 anchor
- `box_parameterization = yolov5`
- `soft_objectness_target = iou`
- `soft_objectness_min = 0.4`

本轮唯一主变量是：

- 将 `encode_target()` 中的 anchor 正样本匹配标准
- 从基于宽高 IoU 的逻辑
- 改为基于宽高比例的 `Shape-Matching`

对应变更说明：

- [stagec_round3c_change_notes.md](/home/lidz/YOLO/yolov0/logs/stagec_round3c_change_notes.md:1)

## 2. 对应实验

- 实验名：
  - `deep_residual_three_box_v5box_softobj_shapematch`
- run_id：
  - `deep_residual_three_box_v5box_softobj_shapematch_20260419_184314`
- 配置：
  - [deep_residual_three_box_v5box_softobj_shapematch.toml](/home/lidz/YOLO/yolov0/configs/deep_residual_three_box_v5box_softobj_shapematch.toml:1)
- 结果：
  - [result.txt](/home/lidz/YOLO/yolov0/logs/records/deep_residual_three_box_v5box_softobj_shapematch_20260419_184314/result.txt:1)
- 元数据：
  - [metadata.json](/home/lidz/YOLO/yolov0/logs/records/deep_residual_three_box_v5box_softobj_shapematch_20260419_184314/metadata.json:1)
- 评估：
  - [internal eval](/home/lidz/YOLO/yolov0/outputs/evaluations/deep_residual_three_box_v5box_softobj_shapematch_eval.json:1)
  - [COCO subset eval](/home/lidz/YOLO/yolov0/outputs/evaluations/deep_residual_three_box_v5box_softobj_shapematch_coco_eval.json:1)

## 3. 训练结果

### 核心训练统计

- 输出形状：
  - `(1, 10, 10, 255)`
- 参数量：
  - `26.56M`
- 最优 epoch：
  - `16`
- 最优验证总损失：
  - `2.585076`
- 最终训练总损失：
  - `2.301005`
- 最终验证总损失：
  - `2.630886`

### 训练过程观测

本轮训练有两个明显特征：

1. `best epoch` 后移到了 `16`
   - 说明 shape-matching 没有造成早期发散，反而让有效学习持续更久。
2. `mean_obj_target` 与 `mean_giou` 全程同步上升
   - 说明三框中的正样本覆盖和回归质量都在稳步改善。

同时，`ignored_count` 仍然接近 `0`，说明这轮真正起作用的是：

- 正样本覆盖变宽
- 而不是 ignore 区域突然增多

## 4. 与前几轮三框版本对比

| 指标 | Round-1 | Round-2 | Round-3B | Round-3C | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| Best Val Total | `2.530998` | `2.640918` | `2.673807` | `2.585076` | 低于 Round-1，但明显优于 Round-2/3B |
| Final Train Total | `2.123002` | `2.376080` | `2.439022` | `2.301005` | 优于 Round-2/3B |
| Final Val Total | `2.586745` | `2.687649` | `2.723669` | `2.630886` | 优于 Round-2/3B |
| Internal mAP@0.5 | `0.021857` | `0.024256` | `0.020722` | `0.023924` | 仅次于 Round-2 |
| Internal Precision | `0.026856` | `0.036024` | `0.028406` | `0.035067` | 接近 Round-2，明显优于 Round-1/3B |
| Internal Recall | `0.199989` | `0.200331` | `0.200198` | `0.201168` | 四轮最高 |
| COCO AP50 | `0.000299` | `0.000286` | `0.000296` | `0.000343` | 四轮最高 |
| COCO AR@100 | `0.001134` | `0.001014` | `0.001090` | `0.001332` | 四轮最高 |

## 5. 与当前单框最强基线对比

当前单框最强基线仍然是：

- `deep_residual_single_box_full_loss`

对比如下：

| 指标 | Stage B 单框残差 | Stage C Round-3C 三框 | 结论 |
| --- | ---: | ---: | --- |
| Best Val Total | `2.351891` | `2.585076` | 三框仍更高 |
| Internal mAP@0.5 | `0.028802` | `0.023924` | 三框仍更低 |
| Internal Precision | `0.046880` | `0.035067` | 三框仍更低 |
| Internal Recall | `0.201682` | `0.201168` | 基本持平 |
| COCO AP50 | `0.000588` | `0.000343` | 三框仍更低 |
| COCO AR@100 | `0.001211` | `0.001332` | 三框首次超过单框 |

## 6. 主要结论

### 6.1 Shape-Matching 是当前 Stage C 最有效的一次三框修正

Round-3C 虽然没有在所有指标上全面超过单框残差基线，但它已经清楚地证明：

- 三框真正缺的不是更大 backbone
- 而是更合理的 anchor 匹配标准

与 Round-1 / Round-2 / Round-3B 相比：

- `COCO AP50` 达到当前三框线最高
- `COCO AR@100` 达到当前三框线最高
- internal recall 也达到当前三框线最高

这说明 shape-matching 的方向是对的。

### 6.2 当前三框线已经从“明显落后”推进到“召回能力更强，但精度仍不足”

和单框残差基线相比，Round-3C 的特点很清楚：

- `Recall / AR@100` 更接近甚至略优
- `Precision / AP50` 仍然偏低

这意味着当前三框线已经开始体现出：

- 更强的覆盖能力

但仍然存在：

- objectness / classification / ranking 还不够准

### 6.3 下一步仍应继续围绕三框机制本身改，而不是立刻扩系统

Round-3C 的结果已经足以说明：

- 当前 Stage C 线值得继续保留
- 但下一步仍然应该是三框机制内部优化

更具体地说，后面最值得继续看的方向是：

- objectness 与框质量的进一步耦合
- crowded cell 下的分配策略
- anchor 使用规则的进一步细化

而不是立刻跳去：

- 更大输入
- 更细网格
- neck
- 多尺度 head

## 7. 当前状态

截至 Round-3C：

- `deep_residual + single-box full loss`
  - 仍然是当前最强实用基线
- `deep_residual + three-box`
  - 已经通过 shape-matching 获得了目前为止最强的三框结果

因此当前阶段的最准确判断是：

> Round-3C 没有彻底翻盘单框基线，但它已经把三框线从“可训练但偏弱”推进到了“召回能力开始体现优势、且官方风格 AP50 继续上升”的状态，说明这条线仍然值得继续优化。
