# Stage C Round 3C 变更说明

本轮实验在 `Round 3B` 的基础上继续推进，但只改一个核心变量：

- 将 Anchor 正样本匹配规则从 `IoU` 改为 `Shape-Matching`

## 1. 保持不变的内容

以下设置全部保持和 `Round 3B` 一致：

- backbone：`deep_residual`
- 输入尺寸：`320 x 320`
- 输出网格：`10 x 10`
- 三框检测头：`num_boxes = 3`
- anchors：`0.052141,0.076305;0.199926,0.288289;0.609263,0.649287`
- box 参数化：`yolov5`
- soft objectness：`IoU soft target + min clamp 0.4`
- full train / full val

## 2. 本轮唯一主变量

在 `data/target_encoder.py` 中：

- 不再用 `anchor_iou` 作为主要正样本候选标尺
- 改为使用宽高比例失配度：
  - `shape_match = max(max(box_w/anchor_w, anchor_w/box_w), max(box_h/anchor_h, anchor_h/box_h))`
- 只要 `shape_match <= 4.0`，就允许 anchor 成为正样本或忽略样本候选

## 3. 设计动机

当前 Stage C 的问题之一是：IoU 匹配过于依赖位置关系，导致 shape 很合适但位置略有偏差的 anchor 无法被激活。

Round 3C 的目标是：

- 让 shape 合适的 anchor 更稳定地获得监督
- 进一步释放三框机制的表达能力
- 观察 internal mAP / Recall 与 COCO 子集 AP50 是否同步改善

## 4. 结果记录要求

本轮 full-run 完成后，必须同步补齐：

- `result.txt`
- `metadata.json`
- internal eval json
- COCO subset eval json
- 可视化图
- 阶段总结
- 专用 Git commit
