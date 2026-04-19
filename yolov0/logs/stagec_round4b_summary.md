# Stage C Round 4B Summary

本文件总结 Stage C 第四轮改进中的第二步：

- `Round 4B: 在 shape-matching 下引入真实生效的 ignore 区间`

## 1. 实验目的

Round 4A 已经证明：

- 收紧 `anchor_shape_ratio`
- 降低 `soft_objectness_min`

能够减少多余预测框，并明显提高 `COCO AP50`。

但 Round 4A 还有一个明显问题：

- `ignored_count` 始终为 `0`

这说明在 `shape-matching` 分支中，当前的 ignore 逻辑实际上没有真正发挥作用。

因此 Round 4B 的目标是：

- 保留 Round 4A 已经验证有效的“收紧三框”方向；
- 在此基础上，建立一个真正工作的 `positive / ignore / negative` 三态分配；
- 观察这是否能够在不完全压缩召回的前提下，改善三框的训练和检测行为。

## 2. 控制变量

以下设置全部保持和 `Round 4A` 一致：

- backbone：`deep_residual`
- 输入尺寸：`320 x 320`
- 输出网格：`10 x 10`
- 三框检测头：`num_boxes = 3`
- anchors：`0.052141,0.076305;0.199926,0.288289;0.609263,0.649287`
- `anchor_match_metric = "shape_ratio"`
- `anchor_shape_ratio = 2.5`
- `soft_objectness_min = 0.05`
- `box_parameterization = "yolov5"`
- full train / full val
- 内部评估协议与 COCO 子集评估协议

本轮唯一主变量是：

- 在 `shape-matching` 分支中增加单独的 `anchor_ignore_shape_ratio = 4.0`
- 使：
  - `shape_ratio <= 2.5`：正样本候选
  - `2.5 < shape_ratio <= 4.0`：ignore
  - `> 4.0`：负样本

## 3. 对应实验

- 实验名：
  - `deep_residual_three_box_v5box_softobj_shapematch_ignore`
- run_id：
  - `deep_residual_three_box_v5box_softobj_shapematch_ignore_20260419_215025`
- 配置：
  - [deep_residual_three_box_v5box_softobj_shapematch_ignore.toml](/home/lidz/YOLO/yolov0/configs/deep_residual_three_box_v5box_softobj_shapematch_ignore.toml:1)
- 结果：
  - [result.txt](/home/lidz/YOLO/yolov0/logs/records/deep_residual_three_box_v5box_softobj_shapematch_ignore_20260419_215025/result.txt:1)
- 元数据：
  - [metadata.json](/home/lidz/YOLO/yolov0/logs/records/deep_residual_three_box_v5box_softobj_shapematch_ignore_20260419_215025/metadata.json:1)
- 评估：
  - [internal eval](/home/lidz/YOLO/yolov0/outputs/evaluations/deep_residual_three_box_v5box_softobj_shapematch_ignore_eval.json:1)
  - [COCO subset eval](/home/lidz/YOLO/yolov0/outputs/evaluations/deep_residual_three_box_v5box_softobj_shapematch_ignore_coco_eval.json:1)

## 4. 训练结果

### 核心训练统计

- 输出形状：
  - `(1, 10, 10, 255)`
- 参数量：
  - `26.56M`
- 最优 epoch：
  - `15`
- 最优验证总损失：
  - `2.626044`
- 最终训练总损失：
  - `2.386118`
- 最终验证总损失：
  - `2.677928`

### 训练过程观测

Round 4B 和前几轮最大的不同，是 `ignore` 终于开始真实工作：

- train `ignored ≈ 2.368`
- val `ignored ≈ 1.867`

这说明本轮改动不再只是“收紧正样本”，而是第一次真正建立了：

- 正样本
- 忽略样本
- 负样本

三个层次。

## 5. 与前一轮 Round 4A 对比

| 指标 | Round 4A | Round 4B | 变化解读 |
| --- | ---: | ---: | --- |
| Best Val Total | `2.620993` | `2.626044` | 基本持平，略差 |
| Best Epoch | `15` | `15` | 一致 |
| Final Train Total | `2.378493` | `2.386118` | 基本一致 |
| Final Val Total | `2.683031` | `2.677928` | 略优 |
| Internal mAP@0.5 | `0.021794` | `0.023020` | 回升 |
| Internal Precision | `0.035762` | `0.033671` | 下降 |
| Internal Recall | `0.199989` | `0.202157` | 上升 |
| Num Predictions | `293968` | `315617` | 明显增加 |
| COCO AP50 | `0.000496` | `0.000432` | 回落 |
| COCO AR@100 | `0.001103` | `0.001175` | 回升 |

## 6. 与 Round 3C 对比

| 指标 | Round 3C | Round 4B | 变化解读 |
| --- | ---: | ---: | --- |
| Internal mAP@0.5 | `0.023924` | `0.023020` | 略低 |
| Internal Precision | `0.035067` | `0.033671` | 略低 |
| Internal Recall | `0.201168` | `0.202157` | 更高 |
| Num Predictions | `301567` | `315617` | 更多 |
| COCO AP50 | `0.000343` | `0.000432` | 明显更高 |
| COCO AR@100 | `0.001332` | `0.001175` | 略低 |

## 7. 与当前单框最强基线对比

当前单框最强基线仍然是：

- `deep_residual_single_box_full_loss`

对比如下：

| 指标 | Stage B 单框残差 | Round 4B 三框 ignore 版 | 结论 |
| --- | ---: | ---: | --- |
| Internal mAP@0.5 | `0.028802` | `0.023020` | 三框仍更低 |
| Internal Precision | `0.046880` | `0.033671` | 三框仍更低 |
| Internal Recall | `0.201682` | `0.202157` | 三框略高 |
| COCO AP50 | `0.000588` | `0.000432` | 三框仍低 |
| COCO AR@100 | `0.001211` | `0.001175` | 基本接近 |

## 8. 主要结论

### 8.1 Round 4B 证明了 ignore 区间确实有意义

本轮最核心的工程收获不是某一个单独指标，而是：

- `ignored_count` 终于显著大于 `0`

这说明 Round 4B 第一次真正把三态分配落到了代码和训练行为里。

### 8.2 Round 4B 更偏向“恢复召回”，而不是继续压精度

和 Round 4A 相比，本轮表现出非常明确的取舍：

- recall 上升
- `AR@100` 上升
- `mAP@0.5` 回升

但同时：

- precision 下降
- `AP50` 从 Round 4A 的高点回落
- 预测框数量重新增加

所以 Round 4B 的作用更像是：

> 在 Round 4A 过度收紧之后，重新找回一部分三框的覆盖能力。

### 8.3 当前最优三框版本仍然是“分目标看待”

如果关注：

- 排序质量
- `COCO AP50`
- 可视化的干净程度

那么当前更优的是：

- `Round 4A`

如果关注：

- 召回
- `AR@100`
- 更完整的候选覆盖

那么 `Round 4B` 是有价值的。

因此，截至 Round 4B，当前三框线的最准确判断是：

> Round 4A 更像“精度导向版本”，Round 4B 更像“召回导向版本”。两者都优于更早的 Stage C 版本，但都还没有全面超过单框残差基线。

## 9. 下一步方向

Round 4B 的结果说明，当前 Stage C 已经进入一个更明确的矛盾：

- 如果收得过紧，`AP50` 提升但召回回落；
- 如果放开 ignore，召回恢复但预测框数量又会增加，精度回退。

这意味着下一步不该再继续大幅放宽或收紧匹配本身，而应该优先改：

- `objectness` 与框质量的耦合
- 多框的排序质量

也就是说，下一步更应该沿着：

- 保留 `shape-matching`
- 保留真实 ignore band
- 继续优化 objectness / scoring quality

这条线继续推进。
