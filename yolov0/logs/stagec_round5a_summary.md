# Stage C Round 5A Summary

本文件总结 Stage C 第五轮改进中的第一步：

- `Round 5A: 让分类分支感知 IoU 质量`

## 1. 实验目的

Round 4A 和 Round 4B 的结果已经把当前三框的主要矛盾暴露出来：

- Round 4A 更偏向精度导向，`AP50` 更高，红框更干净；
- Round 4B 更偏向召回导向，`Recall / AR@100` 更高，但预测框重新增多。

这说明当前问题不再只是“是否能找到目标”，而是：

- 多框下的排序质量还不够好；
- 一些回归质量较差的框，仍然可能拿到偏高的分类分数；
- `score = obj * cls` 的最终排序仍然会被这些劣质框扰动。

因此 Round 5A 的目标是：

- 保留 Round 4B 已经建立起来的 `shape-matching + ignore band`；
- 不改 backbone、不改 head、不改 matcher 主体；
- 只修改分类监督，让分类目标也随当前预测框质量变化；
- 观察这是否能在保住召回的同时改善排序质量。

## 2. 控制变量

以下设置全部保持和 `Round 4B` 一致：

- backbone：`deep_residual`
- 输入尺寸：`320 x 320`
- 输出网格：`10 x 10`
- 三框检测头：`num_boxes = 3`
- anchors：`0.052141,0.076305;0.199926,0.288289;0.609263,0.649287`
- `anchor_match_metric = "shape_ratio"`
- `anchor_shape_ratio = 2.5`
- `anchor_ignore_shape_ratio = 4.0`
- `box_parameterization = "yolov5"`
- `soft_objectness_target = "iou"`
- `soft_objectness_min = 0.05`
- full train / full val
- 内部评估协议与 COCO 子集评估协议

本轮唯一主变量是：

- 在分类分支中，将 one-hot 正样本目标从硬标签 `1.0` 改成当前预测框质量：
  - `soft_classification_target = "iou"`

## 3. 对应实验

- 实验名：
  - `deep_residual_three_box_qualitycls`
- run_id：
  - `deep_residual_three_box_qualitycls_20260420_001813`
- 配置：
  - [deep_residual_three_box_qualitycls.toml](/home/lidz/YOLO/yolov0/configs/deep_residual_three_box_qualitycls.toml:1)
- 结果：
  - [result.txt](/home/lidz/YOLO/yolov0/logs/records/deep_residual_three_box_qualitycls_20260420_001813/result.txt:1)
- 元数据：
  - [metadata.json](/home/lidz/YOLO/yolov0/logs/records/deep_residual_three_box_qualitycls_20260420_001813/metadata.json:1)
- 评估：
  - [internal eval](/home/lidz/YOLO/yolov0/outputs/evaluations/deep_residual_three_box_qualitycls_eval.json:1)
  - [COCO subset eval](/home/lidz/YOLO/yolov0/outputs/evaluations/deep_residual_three_box_qualitycls_coco_eval.json:1)

## 4. 训练结果

### 核心训练统计

- 输出形状：
  - `(1, 10, 10, 255)`
- 参数量：
  - `26.56M`
- 最优 epoch：
  - `16`
- 最优验证总损失：
  - `2.621144`
- 最终训练总损失：
  - `2.371728`
- 最终验证总损失：
  - `2.669160`

### 训练过程观测

Round 5A 的一个直接现象是：

- `mean_cls_target` 不再是固定的 `1.0`
- 而是跟随当前正样本框质量变化

从结果文件可以看到：

- train `mean_cls_target` 从约 `0.46` 上升到约 `0.64`
- val `mean_cls_target` 从约 `0.49` 上升到约 `0.64`

这说明本轮改动并没有让分类分支“失去方向”，反而让分类目标逐步和框质量同步抬升。

## 5. 与前一轮 Round 4B 对比

| 指标 | Round 4B | Round 5A | 变化解读 |
| --- | ---: | ---: | --- |
| Best Val Total | `2.626044` | `2.621144` | 略优 |
| Best Epoch | `15` | `16` | 稍晚一轮 |
| Final Train Total | `2.386118` | `2.371728` | 略优 |
| Final Val Total | `2.677928` | `2.669160` | 略优 |
| Internal mAP@0.5 | `0.023020` | `0.019629` | 明显下降 |
| Internal Precision | `0.033671` | `0.056758` | 明显提升 |
| Internal Recall | `0.202157` | `0.176362` | 明显下降 |
| Num Predictions | `315617` | `163343` | 大幅减少 |
| COCO AP50 | `0.000432` | `0.000415` | 略降 |
| COCO AR@100 | `0.001175` | `0.000782` | 明显下降 |

## 6. 与 Round 4A 对比

| 指标 | Round 4A | Round 5A | 变化解读 |
| --- | ---: | ---: | --- |
| Internal mAP@0.5 | `0.021794` | `0.019629` | 略低 |
| Internal Precision | `0.035762` | `0.056758` | 更高 |
| Internal Recall | `0.199989` | `0.176362` | 更低 |
| Num Predictions | `293968` | `163343` | 明显更少 |
| COCO AP50 | `0.000496` | `0.000415` | 低于 Round 4A |
| COCO AR@100 | `0.001103` | `0.000782` | 更低 |

## 7. 与当前单框最强基线对比

当前单框最强基线仍然是：

- `deep_residual_single_box_full_loss`

对比如下：

| 指标 | Stage B 单框残差 | Round 5A 三框质量分类版 | 结论 |
| --- | ---: | ---: | --- |
| Internal mAP@0.5 | `0.028802` | `0.019629` | 三框仍更低 |
| Internal Precision | `0.046880` | `0.056758` | 三框更高 |
| Internal Recall | `0.201682` | `0.176362` | 三框明显更低 |
| COCO AP50 | `0.000588` | `0.000415` | 三框仍低 |
| COCO AR@100 | `0.001211` | `0.000782` | 三框仍低 |

## 8. 主要结论

### 8.1 质量感知分类确实压住了“多余框”

本轮最明显的直接收益是：

- `num_predictions` 从 `315617` 大幅降到 `163343`
- internal precision 从 `0.033671` 提升到 `0.056758`

这说明让分类目标感知 IoU 质量，确实能显著抑制劣质框的最终分数。

### 8.2 但本轮收得过头，召回下降过大

本轮同时也带来了明显代价：

- internal recall 从 `0.202157` 降到 `0.176362`
- COCO `AR@100` 从 `0.001175` 降到 `0.000782`

这说明当前 Round 5A 的分类质量约束虽然能“杀掉烂框”，但也把一部分原本有潜力的边缘框一起压掉了。

### 8.3 Round 5A 更像“排序约束验证”，而不是最终可用版本

截至本轮，最准确的判断是：

- 质量感知分类这个方向是有效的；
- 它证明了三框当前确实存在“分类分数偏虚高”的问题；
- 但当前这版做法太强，导致召回回落明显；
- 因此更适合作为下一步结构改进（解耦头）的依据，而不是当前三框的最终版本。

## 9. 下一步方向

Round 5A 已经说明：

- 三框当前最主要的矛盾之一，确实是分类与回归排序质量不足；
- 但仅靠在原共享检测头上直接施加质量感知分类，会把召回压得过低。

因此下一步更合理的方向是：

- 保留 Round 4B 的 `shape-matching + ignore band`
- 引入 `Decoupled Head`
- 让分类分支与回归/obj 分支不再完全共享同一套最终卷积特征

也就是说，Round 5B 的目标不是继续收紧 loss，而是：

> 在不放弃质量感知路线的前提下，先从结构上减轻分类与回归的冲突。
