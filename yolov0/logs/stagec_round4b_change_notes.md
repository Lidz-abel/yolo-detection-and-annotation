# Stage C Round 4B 变更说明

本轮实验在 `Round 4A` 的基础上继续推进，但只做一项核心逻辑修正：

- 为 `shape-matching` 分支引入真正独立的 `ignore_ratio`

## 1. 保持不变的内容

以下设置全部保持和 `Round 4A` 一致：

- backbone：`deep_residual`
- 输入尺寸：`320 x 320`
- 输出网格：`10 x 10`
- 三框检测头：`num_boxes = 3`
- anchors：`0.052141,0.076305;0.199926,0.288289;0.609263,0.649287`
- `anchor_match_metric = "shape_ratio"`
- `anchor_shape_ratio = 2.5`
- `box_parameterization = "yolov5"`
- `soft_objectness_target = "iou"`
- `soft_objectness_min = 0.05`
- full train / full val

## 2. 本轮唯一主变量

本轮新增并启用：

- `anchor_ignore_shape_ratio = 4.0`

新的三态规则是：

- `shape_ratio <= 2.5`
  - 正样本候选
- `2.5 < shape_ratio <= 4.0`
  - ignore 样本
- `shape_ratio > 4.0`
  - 负样本

## 3. 设计动机

在 `Round 4A` 中，虽然通过收紧 `anchor_shape_ratio` 和下调
`soft_objectness_min`，已经显著改善了 `COCO AP50`，但训练记录仍然显示：

- `ignored_count = 0`

这说明当前 `shape-matching` 分支里的 ignore 条件实际上没有真正发挥作用。

因此本轮的目标是：

- 让正样本和负样本之间存在真实的缓冲区；
- 避免“勉强匹配但不该当正样本”的 anchor 被直接推到负样本惩罚；
- 进一步压低多余框，同时尽量保住 Round 4A 已经获得的精度收益。

## 4. 结果记录要求

本轮 full-run 完成后，必须同步补齐：

- `result.txt`
- `metadata.json`
- internal eval json
- COCO subset eval json
- 可视化图
- 阶段总结
- 专用 Git commit
