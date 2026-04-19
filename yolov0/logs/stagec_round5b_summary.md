# Stage C Round 5B Summary

本文件总结 Stage C 第五轮改进中的第二步：

- `Round 5B: 在 Round 5A 基础上引入 Decoupled Head`

## 1. 实验目的

Round 5A 已经证明：

- 让分类监督感知 IoU 质量，能够显著压低低质量框的最终分数；
- 但在当前共享检测头上，这种约束也把召回压得过低。

这说明当前三框的一个核心冲突是：

- 分类分支与回归 / objectness 分支仍然共享同一套头部特征；
- 分类追求语义判别，回归追求几何精度；
- 二者在共享 head 中相互牵制，导致排序质量和召回难以兼顾。

因此 Round 5B 的目标是：

- 保留 Round 5A 已经验证有效的“质量感知分类”；
- 保留 Round 4B 的 `shape-matching + ignore band`；
- 不改 backbone，不改 matcher，不改 box 参数化；
- 只把检测头改成解耦结构，让 `cls` 与 `box+obj` 使用不同分支。

## 2. 控制变量

以下设置全部保持和 `Round 5A` 一致：

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
- `soft_classification_target = "iou"`
- full train / full val
- 内部评估协议与 COCO 子集评估协议

本轮唯一主变量是：

- `head_type = "decoupled"`

即检测头从共享输出改成：

- 一条分类分支
- 一条回归 + objectness 分支

## 3. 对应实验

- 实验名：
  - `deep_residual_three_box_qualitycls_decoupled`
- run_id：
  - `deep_residual_three_box_qualitycls_decoupled_20260420_014124`
- 配置：
  - [deep_residual_three_box_qualitycls_decoupled.toml](/home/lidz/YOLO/yolov0/configs/deep_residual_three_box_qualitycls_decoupled.toml:1)
- 结果：
  - [result.txt](/home/lidz/YOLO/yolov0/logs/records/deep_residual_three_box_qualitycls_decoupled_20260420_014124/result.txt:1)
- 元数据：
  - [metadata.json](/home/lidz/YOLO/yolov0/logs/records/deep_residual_three_box_qualitycls_decoupled_20260420_014124/metadata.json:1)
- 评估：
  - [internal eval](/home/lidz/YOLO/yolov0/outputs/evaluations/deep_residual_three_box_qualitycls_decoupled_eval.json:1)
  - [COCO subset eval](/home/lidz/YOLO/yolov0/outputs/evaluations/deep_residual_three_box_qualitycls_decoupled_coco_eval.json:1)

## 4. 训练结果

### 核心训练统计

- 输出形状：
  - `(1, 10, 10, 255)`
- 参数量：
  - `28.92M`
- 最优 epoch：
  - `16`
- 最优验证总损失：
  - `2.597307`
- 最终训练总损失：
  - `2.470387`
- 最终验证总损失：
  - `2.629688`

### 训练过程观测

Round 5B 的一个非常明显的现象是：

- train / val loss 曲线下降更平滑
- best val loss 进一步优于 Round 5A
- 分类目标仍然保持质量感知，但没有再明显压缩召回

从结果文件可以看到：

- val total 从 `3.5359` 稳定降到 `2.5973`
- val `obj_target` 持续从 `0.48` 抬升到约 `0.64`
- val `cls` loss 基本稳定在 `0.036` 左右，没有崩溃

## 5. 与前一轮 Round 5A 对比

| 指标 | Round 5A | Round 5B | 变化解读 |
| --- | ---: | ---: | --- |
| Best Val Total | `2.621144` | `2.597307` | 进一步改善 |
| Best Epoch | `16` | `16` | 一致 |
| Final Train Total | `2.371728` | `2.470387` | 略高，但不影响最佳点 |
| Final Val Total | `2.669160` | `2.629688` | 更好 |
| Internal mAP@0.5 | `0.019629` | `0.077865` | 大幅提升 |
| Internal Precision | `0.056758` | `0.053212` | 略低 |
| Internal Recall | `0.176362` | `0.259645` | 大幅提升 |
| Num Predictions | `163343` | `256503` | 回升，但仍低于 Round 4B |
| COCO AP50 | `0.000415` | `0.000814` | 大幅提升 |
| COCO AR@100 | `0.000782` | `0.002500` | 大幅提升 |

## 6. 与 Round 4B 对比

| 指标 | Round 4B | Round 5B | 变化解读 |
| --- | ---: | ---: | --- |
| Internal mAP@0.5 | `0.023020` | `0.077865` | 大幅提升 |
| Internal Precision | `0.033671` | `0.053212` | 明显更高 |
| Internal Recall | `0.202157` | `0.259645` | 明显更高 |
| Num Predictions | `315617` | `256503` | 更少 |
| COCO AP50 | `0.000432` | `0.000814` | 明显更高 |
| COCO AR@100 | `0.001175` | `0.002500` | 明显更高 |

## 7. 与当前单框最强基线对比

当前单框最强基线仍然是：

- `deep_residual_single_box_full_loss`

对比如下：

| 指标 | Stage B 单框残差 | Round 5B 三框解耦头 | 结论 |
| --- | ---: | ---: | --- |
| Internal mAP@0.5 | `0.028802` | `0.077865` | 三框显著更高 |
| Internal Precision | `0.046880` | `0.053212` | 三框更高 |
| Internal Recall | `0.201682` | `0.259645` | 三框明显更高 |
| COCO AP50 | `0.000588` | `0.000814` | 三框更高 |
| COCO AR@100 | `0.001211` | `0.002500` | 三框明显更高 |

## 8. 主要结论

### 8.1 解耦头是当前 Stage C 的关键突破点

Round 5B 的结果非常明确：

- `AP50`
- `AR@100`
- internal `mAP@0.5`
- internal `recall`

全部明显优于 Round 5A 与 Round 4B。

这说明：

- 当前三框线的核心瓶颈之一，确实不是 matcher 本身，而是 head 中分类与回归的耦合冲突。

### 8.2 Round 5B 首次让三框线全面超过单框残差基线

截至本轮：

- 三框版本不仅在召回上更强
- 在更严格的 COCO 子集 `AP50` 上也已经超过 Stage B 单框残差基线

这意味着当前 Stage C 已经从：

- “能训练但不够强”

推进到：

- “在当前 checkpoint 6 设定下，已经具备明确工程价值”

### 8.3 当前最佳三框版本已经从“收紧匹配”转为“结构改进”

Round 4A / 4B 的作用主要是：

- 建立更合理的匹配与 ignore 逻辑
- 找到精度与召回的平衡矛盾

而 Round 5B 的结果表明：

- 真正让三框跨过当前阶段门槛的，不只是继续调 matcher
- 而是通过 `Decoupled Head` 让分类与回归能各自学习更合适的表示

## 9. 下一步方向

Round 5B 已经证明：

- `shape-matching + ignore band + quality-aware cls + decoupled head`
- 是当前 Stage C 最强的基础版本

因此下一步不应回头否定这条线，而应该在它之上继续推进：

- 进入 `Round 5C`
- 尝试把静态 shape-matching 升级成基于预测质量的动态分配

也就是说，当前最合理的下一步是：

> 以 Round 5B 为新三框基线，继续推进 `Dynamic Assignment`，观察是否还能在不破坏当前排序质量的前提下进一步提升正样本分配效率。
