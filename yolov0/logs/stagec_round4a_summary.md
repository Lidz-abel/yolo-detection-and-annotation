# Stage C Round 4A Summary

本文件总结 Stage C 第四轮改进中的第一步：

- `Round 4A: 纯配置收紧`

## 1. 实验目的

Round 4A 的目标不是继续放宽三框，而是主动收紧当前已经被激活的
`shape-matching` 三框机制，验证下面两个判断是否成立：

- `anchor_shape_ratio = 4.0` 是否过于宽松，导致过多勉强匹配的 anchor
  被收为正样本；
- `soft_objectness_min = 0.4` 是否过高，导致回归质量很差的框也能获得
  过高的 objectness 监督，进而在可视化和评估中形成大量“僵尸框”。

因此，本轮重点不是追求更高召回，而是验证：

- 是否能在保持训练稳定的前提下，
- 压低预测框数量，
- 提升排序质量和 `AP50`。

## 2. 控制变量

以下设置全部保持和 `Round 3C` 一致：

- backbone：`deep_residual`
- 输入尺寸：`320 x 320`
- 输出网格：`10 x 10`
- 三框检测头：`num_boxes = 3`
- anchors：`0.052141,0.076305;0.199926,0.288289;0.609263,0.649287`
- `anchor_match_metric = "shape_ratio"`
- `box_parameterization = "yolov5"`
- full train / full val
- 内部评估协议与 COCO 子集评估协议

本轮唯一主变量是配置收紧：

- `anchor_shape_ratio: 4.0 -> 2.5`
- `soft_objectness_min: 0.4 -> 0.05`
- `visualization.score_threshold: 0.05 -> 0.25`

其中最后一项只影响可视化，不影响正式评估。

## 3. 对应实验

- 实验名：
  - `deep_residual_three_box_v5box_softobj_shapematch_tight`
- run_id：
  - `deep_residual_three_box_v5box_softobj_shapematch_tight_20260419_201922`
- 配置：
  - [deep_residual_three_box_v5box_softobj_shapematch_tight.toml](/home/lidz/YOLO/yolov0/configs/deep_residual_three_box_v5box_softobj_shapematch_tight.toml:1)
- 结果：
  - [result.txt](/home/lidz/YOLO/yolov0/logs/records/deep_residual_three_box_v5box_softobj_shapematch_tight_20260419_201922/result.txt:1)
- 元数据：
  - [metadata.json](/home/lidz/YOLO/yolov0/logs/records/deep_residual_three_box_v5box_softobj_shapematch_tight_20260419_201922/metadata.json:1)
- 评估：
  - [internal eval](/home/lidz/YOLO/yolov0/outputs/evaluations/deep_residual_three_box_v5box_softobj_shapematch_tight_eval.json:1)
  - [COCO subset eval](/home/lidz/YOLO/yolov0/outputs/evaluations/deep_residual_three_box_v5box_softobj_shapematch_tight_coco_eval.json:1)

## 4. 训练结果

### 核心训练统计

- 输出形状：
  - `(1, 10, 10, 255)`
- 参数量：
  - `26.56M`
- 最优 epoch：
  - `15`
- 最优验证总损失：
  - `2.620993`
- 最终训练总损失：
  - `2.378493`
- 最终验证总损失：
  - `2.683031`

### 训练过程观测

Round 4A 的训练表现有两个明确特征：

1. 验证总损失持续下降到 `epoch 15`
   - 说明“收紧”没有让训练塌掉，反而让优化更稳。
2. `obj` 分量持续下降
   - 说明更低的 `soft_objectness_min` 没有造成正样本 objectness 无法学习，
     而是帮助模型逐步压低过强的置信度。

同时，`ignored_count` 仍然保持 `0.0`，说明本轮真正发生变化的是：

- 正样本筛选变严；
- 低质量正样本的 objectness 监督变弱；

而不是 ignore 区域开始发挥作用。

## 5. 与前一轮 Round 3C 对比

| 指标 | Round 3C | Round 4A | 变化解读 |
| --- | ---: | ---: | --- |
| Best Val Total | `2.585076` | `2.620993` | 略差，但仍处在相近区间 |
| Best Epoch | `16` | `15` | 基本一致 |
| Final Train Total | `2.301005` | `2.378493` | 收紧后训练更保守 |
| Final Val Total | `2.630886` | `2.683031` | 略差 |
| Internal mAP@0.5 | `0.023924` | `0.021794` | 小幅下降 |
| Internal Precision | `0.035067` | `0.035762` | 小幅上升 |
| Internal Recall | `0.201168` | `0.199989` | 小幅下降 |
| Num Predictions | `301567` | `293968` | 明显减少 |
| COCO AP50 | `0.000343` | `0.000496` | 明显提升 |
| COCO AR@100 | `0.001332` | `0.001103` | 有所回落 |

## 6. 与当前单框最强基线对比

当前单框最强基线仍然是：

- `deep_residual_single_box_full_loss`

对比如下：

| 指标 | Stage B 单框残差 | Round 4A 三框收紧版 | 结论 |
| --- | ---: | ---: | --- |
| Best Val Total | `2.351891` | `2.620993` | 三框仍更高 |
| Internal mAP@0.5 | `0.028802` | `0.021794` | 三框仍更低 |
| Internal Precision | `0.046880` | `0.035762` | 三框仍更低 |
| Internal Recall | `0.201682` | `0.199989` | 基本接近 |
| COCO AP50 | `0.000588` | `0.000496` | 三框已明显接近 |
| COCO AR@100 | `0.001211` | `0.001103` | 三框本轮略低 |

## 7. 主要结论

### 7.1 Round 4A 成功压低了多余框数量

`num_predictions` 从 `301567` 降到 `293968`，说明本轮“收紧”不是空想，
而是真实地减少了输出框数量。

结合可视化目标来看，这说明：

- 一物多框问题开始被压制；
- 低质量框不再像 `Round 3C` 那样轻易通过。

### 7.2 Round 4A 的核心收益体现在精度与排序质量，而不是召回

虽然：

- internal `mAP@0.5` 略降；
- internal `recall` 略降；
- COCO `AR@100` 回落；

但：

- internal `precision` 上升；
- COCO `AP50` 从 `0.000343` 提升到 `0.000496`；

这说明本轮更像是在做一件对 Stage C 现在非常重要的事：

> 用更少的框，换更高的命中质量。

### 7.3 这轮结果支持继续沿“收紧三框”方向推进

Round 4A 给出的信号非常明确：

- 当前三框线已经不是“框不出来”；
- 当前问题主要是“框太多、排序还不够好”；
- 收紧 shape ratio 和 objectness 下限后，`AP50` 的确开始上升。

因此下一步最合理的方向不是回头改 backbone，而是继续围绕：

- `ignore` 逻辑修正；
- crowded cell 下的分配精细化；

继续收紧三框机制。

## 8. 当前状态

截至 Round 4A：

- `deep_residual + single-box full loss`
  - 仍然是当前最强实用基线
- `deep_residual + three-box + shape-match + tight`
  - 是当前最接近单框残差基线的三框版本之一，
  - 并且在 `COCO AP50` 上取得了三框线当前最佳结果

因此当前阶段最准确的判断是：

> Round 4A 虽然没有全面超过 Round 3C，但它第一次明确证明：通过收紧三框的正样本与 objectness 机制，可以减少多余预测框，并显著提升 COCO `AP50`。这说明 Stage C 的下一步应继续沿“控制多框泛滥、提升排序质量”的方向推进。
