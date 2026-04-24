# Stage C Round 6C Summary

本文件总结 Stage C 第六轮改进中的第三步：

- `Round 6C: 在 Round 5C 基础上收紧 Dynamic Assignment`

## 1. 实验目的

Round 5C 已经把三框版本推进到了当前最强的召回与 AP50 水平，但同时也留下了：

- `num_predictions` 偏高
- 套娃框仍然较多
- 排序质量提升慢于覆盖能力提升

因此 Round 6C 的目标是：

- 保留 Round 5C 已经验证有效的动态分配框架；
- 不改 backbone、不改 head、不改质量感知分类；
- 只通过收紧 `dynamic_topk`，减少每个 GT 扩张出的正样本槽位数量；
- 观察是否能在不明显牺牲召回的前提下改善 AP50 和可视化质量。

## 2. 控制变量

以下设置全部保持和 Round 5C 一致：

- backbone：`deep_residual`
- 输入尺寸：`320 x 320`
- 输出网格：`10 x 10`
- 三框检测头：`num_boxes = 3`
- `head_type = "decoupled"`
- `assignment_strategy = "dynamic_cost"`
- `dynamic_center_radius = 1`
- `dynamic_box_cost = 3.0`
- `dynamic_cls_cost = 1.0`
- `dynamic_ignore_iou = 0.5`
- quality-aware classification
- full train / full val
- 内部评估协议与 COCO 子集评估协议

本轮唯一主变量是：

- `dynamic_topk: 2 -> 1`

## 3. 对应实验

- 实验名：
  - `deep_residual_three_box_dynamicassign_topk1`
- run_id：
  - `deep_residual_three_box_dynamicassign_topk1_20260423_210757`
- 配置：
  - [deep_residual_three_box_dynamicassign_topk1.toml](/home/lidz/YOLO/yolov0/configs/deep_residual_three_box_dynamicassign_topk1.toml:1)
- 结果：
  - [result.txt](/home/lidz/YOLO/yolov0/logs/records/deep_residual_three_box_dynamicassign_topk1_20260423_210757/result.txt:1)
- 元数据：
  - [metadata.json](/home/lidz/YOLO/yolov0/logs/records/deep_residual_three_box_dynamicassign_topk1_20260423_210757/metadata.json:1)
- 评估：
  - [internal eval](/home/lidz/YOLO/yolov0/outputs/evaluations/deep_residual_three_box_dynamicassign_topk1_eval.json:1)
  - [COCO subset eval](/home/lidz/YOLO/yolov0/outputs/evaluations/deep_residual_three_box_dynamicassign_topk1_coco_eval.json:1)

## 4. 训练结果

- 输出形状：
  - `(1, 10, 10, 255)`
- 参数量：
  - `28.92M`
- 最优 epoch：
  - `20`
- 最优验证总损失：
  - `2.407872`
- 最终训练总损失：
  - `2.407872`
- 最终验证总损失：
  - `2.407872`

## 5. 与 Round 5C 对比

| 指标 | Round 5C | Round 6C | 变化解读 |
| --- | ---: | ---: | --- |
| Best Val Total | `2.290735` | `2.407872` | 待本轮结论解释 |
| Internal mAP@0.5 | `0.086128` | `0.057019` | 待本轮结论解释 |
| Internal Precision | `0.048140` | `0.031924` | 待本轮结论解释 |
| Internal Recall | `0.306746` | `0.270526` | 待本轮结论解释 |
| Num Predictions | `334960` | `445458` | 候选框数量变化 |
| COCO AP50 | `0.001150` | `0.000854` | 待本轮结论解释 |
| COCO AR@100 | `0.003097` | `0.003093` | 待本轮结论解释 |

## 6. 主要结论

- internal `mAP@0.5 = 0.057019`
- internal `precision = 0.031924`
- internal `recall = 0.270526`
- COCO `AP50 = 0.000854`
- COCO `AR@100 = 0.003093`

Round 6C 的核心问题是：

- `dynamic_topk = 1` 是否能降低套娃框和多余预测；
- 同时尽量保住 Round 5C 已建立起来的召回优势。

如果这轮能提高 AP50 且不明显损伤 AR@100，那么后续就可以继续沿着动态分配收紧的方向做更轻量微调；否则应当保留 Round 5C 作为当前动态分配主基线。
