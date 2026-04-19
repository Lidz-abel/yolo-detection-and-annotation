# Stage C Round 4A 变更说明

本轮实验在 `Round 3C` 的基础上继续推进，但只做“纯配置收紧”，不改核心代码。

## 1. 保持不变的内容

以下设置全部保持和 `Round 3C` 一致：

- backbone：`deep_residual`
- 输入尺寸：`320 x 320`
- 输出网格：`10 x 10`
- 三框检测头：`num_boxes = 3`
- anchors：`0.052141,0.076305;0.199926,0.288289;0.609263,0.649287`
- `anchor_match_metric = "shape_ratio"`
- `box_parameterization = "yolov5"`
- full train / full val

## 2. 本轮唯一主变量

本轮只收紧三个配置项：

1. `anchor_shape_ratio: 4.0 -> 2.5`
   - 目标：减少“形状勉强匹配”的 anchor 被收为正样本。
2. `soft_objectness_min: 0.4 -> 0.05`
   - 目标：剥夺回归质量很差的框的高置信度伪装。
3. `visualization.score_threshold: 0.05 -> 0.25`
   - 目标：让可视化更真实地反映“高分预测框”，避免画图阶段被低分框淹没。

## 3. 设计动机

Round 3C 之后，三框线已经体现出召回优势，但可视化里仍然存在：

- 一物多框
- 套娃框
- 预测框数量显著多于 GT

Round 4A 的目标不是继续扩系统，而是验证：

- 更严格的 shape ratio
- 更低的 objectness floor

能否在不牺牲过多召回的情况下，提高精度和排序质量。

## 4. 结果记录要求

本轮 full-run 完成后，必须同步补齐：

- `result.txt`
- `metadata.json`
- internal eval json
- COCO subset eval json
- 可视化图
- 阶段总结
- 专用 Git commit
