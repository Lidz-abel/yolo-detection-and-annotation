# Stage C Round 5C Summary

本文件总结 Stage C 第五轮改进中的第三步：

- `Round 5C: 在 Round 5B 基础上引入 Dynamic Assignment`

## 1. 实验目的

Round 5B 已经证明：

- 质量感知分类和解耦检测头可以同时提升 `AP50` 和 `AR@100`；
- 当前三框版本第一次在更严格的 COCO 子集评估上超过了单框残差基线；
- 但正样本分配仍然依赖静态的 `shape-matching + ignore band`。

因此 Round 5C 的目标是：

- 保留 `Round 5B` 已经验证有效的 backbone、loss 与 head 设计；
- 不再只依赖静态 anchor 先验；
- 在训练时利用当前预测结果，动态选择代价最低的候选槽位作为正样本；
- 观察动态分配是否还能继续提升 `AP50 / AR@100 / internal mAP@0.5`。

## 2. 控制变量

以下设置全部保持和 `Round 5B` 一致：

- backbone：`deep_residual`
- 输入尺寸：`320 x 320`
- 输出网格：`10 x 10`
- 三框检测头：`num_boxes = 3`
- anchors：`0.052141,0.076305;0.199926,0.288289;0.609263,0.649287`
- `box_parameterization = "yolov5"`
- `soft_objectness_target = "iou"`
- `soft_objectness_min = 0.05`
- `soft_classification_target = "iou"`
- `head_type = "decoupled"`
- full train / full val
- 内部评估协议与 COCO 子集评估协议

本轮唯一主变量是：

- `assignment_strategy = "dynamic_cost"`

以及与之配套的：

- `dynamic_topk = 2`
- `dynamic_center_radius = 1`
- `dynamic_box_cost = 3.0`
- `dynamic_cls_cost = 1.0`
- `dynamic_ignore_iou = 0.5`

## 3. 对应实验

- 实验名：
  - `deep_residual_three_box_dynamicassign`
- run_id：
  - `deep_residual_three_box_dynamicassign_20260420_030501`
- 配置：
  - [deep_residual_three_box_dynamicassign.toml](/home/lidz/YOLO/yolov0/configs/deep_residual_three_box_dynamicassign.toml:1)
- 结果：
  - [result.txt](/home/lidz/YOLO/yolov0/logs/records/deep_residual_three_box_dynamicassign_20260420_030501/result.txt:1)
- 元数据：
  - [metadata.json](/home/lidz/YOLO/yolov0/logs/records/deep_residual_three_box_dynamicassign_20260420_030501/metadata.json:1)
- 评估：
  - [internal eval](/home/lidz/YOLO/yolov0/outputs/evaluations/deep_residual_three_box_dynamicassign_eval.json:1)
  - [COCO subset eval](/home/lidz/YOLO/yolov0/outputs/evaluations/deep_residual_three_box_dynamicassign_coco_eval.json:1)

## 4. 训练结果

### 核心训练统计

- 输出形状：
  - `(1, 10, 10, 255)`
- 参数量：
  - `28.92M`
- 最优 epoch：
  - `20`
- 最优验证总损失：
  - `2.290735`
- 最终训练总损失：
  - `2.582137`
- 最终验证总损失：
  - `2.290735`

### 训练过程观测

动态分配显著抬高了单步计算成本，但训练过程本身保持稳定：

- train total 从 `3.9827` 降到 `2.5821`
- val total 从 `3.3808` 降到 `2.2907`
- `ignored` 数从训练首轮的 `13.52` 逐步抬升到 `33.26`
- `pos_cells` 也从约 `3.78` 抬升到 `4.28`

这说明动态分配确实显著扩大了有效监督范围，但代价是：

- 单 epoch 时间上升到约 `1737s`
- 完整 20 epoch 训练约为 `9.6` 小时级别

## 5. 与前一轮 Round 5B 对比

| 指标 | Round 5B | Round 5C | 变化解读 |
| --- | ---: | ---: | --- |
| Best Val Total | `2.597307` | `2.290735` | 明显下降 |
| Best Epoch | `16` | `20` | 更晚达到最优 |
| Internal mAP@0.5 | `0.077865` | `0.086128` | 继续提升 |
| Internal Precision | `0.053212` | `0.048140` | 略降 |
| Internal Recall | `0.259645` | `0.306746` | 明显提升 |
| Num Predictions | `256503` | `334960` | 明显增多 |
| COCO AP50 | `0.000814` | `0.001150` | 继续提升 |
| COCO AR@100 | `0.002500` | `0.003097` | 继续提升 |
| FPS | `109.08` | `98.41` | 推理略慢 |

## 6. 与当前单框最强基线对比

当前单框最强基线仍然是：

- `deep_residual_single_box_full_loss`

对比如下：

| 指标 | Stage B 单框残差 | Round 5C 动态分配三框 | 结论 |
| --- | ---: | ---: | --- |
| Internal mAP@0.5 | `0.028802` | `0.086128` | 三框显著更高 |
| Internal Precision | `0.046880` | `0.048140` | 三框略高 |
| Internal Recall | `0.201682` | `0.306746` | 三框明显更高 |
| COCO AP50 | `0.000588` | `0.001150` | 三框更高 |
| COCO AR@100 | `0.001211` | `0.003097` | 三框显著更高 |

## 7. 主要结论

### 7.1 动态分配继续提高了三框版本的上限

Round 5C 在更严格的 COCO 子集指标上继续提升：

- `AP50 = 0.001150`
- `AR@100 = 0.003097`

同时 internal 指标也同步提高：

- `mAP@0.5 = 0.086128`
- `recall = 0.306746`

这说明动态分配确实让三框线继续释放了更多潜力。

### 7.2 代价是预测数量进一步增多，精度改善不如召回改善明显

Round 5C 的新问题也很清楚：

- `num_predictions` 增长到了 `334960`
- `precision` 相比 Round 5B 反而略有下降

也就是说，动态分配进一步强化了“覆盖能力”，但没有同步把排序质量拉到同等幅度。

### 7.3 当前最强三框版本已经从静态匹配推进到动态分配

截至本轮：

- `Round 4B` 解决了静态 `shape-matching + ignore band`
- `Round 5A` 引入了质量感知分类
- `Round 5B` 通过 `Decoupled Head` 打开了排序质量瓶颈
- `Round 5C` 则进一步证明：动态分配可以继续提升三框的检测能力

因此当前最强三框基线已经变为：

- `deep_residual + three-box + quality-aware cls + decoupled head + dynamic assignment`

## 8. 下一步建议

Round 5C 的结论不是继续回头改 backbone，而是：

- 保留当前 `dynamic assignment` 版本作为新的三框最强基线；
- 下一步优先处理“预测框数量偏多、排序质量仍可加强”的问题；
- 重点方向应当转向：
  - scoring 公式
  - objectness / cls 的进一步耦合
  - 或更轻量的 dynamic cost 微调

如果后续要进入 Stage D，多尺度也应当建立在当前 Round 5C 的强基线之上，而不是回退到更早的静态匹配版本。
