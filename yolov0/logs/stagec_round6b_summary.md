# Stage C Round 6B Summary

本文件总结 Stage C 第六轮改进中的第二步：

- `Round 6B: 在 Round 5C 训练基线上引入 varifocal-style classification`

## 1. 实验目的

在 `Round 5C` 中，三框线已经取得当前最强的训练侧结果，但仍存在：

- 预测框数量偏多
- 排序质量不稳定
- `AP50` 与 `precision` 仍有提升空间

因此 `Round 6B` 不改 backbone，不改动态分配主框架，只在分类损失上继续增强质量感知：

- 将 `cls_loss_mode` 从普通 `bce` 改为 `varifocal`
- 保留 `soft_classification_target = iou`

目标是：

- 保住 `Round 5C` 的覆盖能力
- 进一步提升排序质量

## 2. 控制变量

以下内容保持与 `Round 5C` 一致：

- backbone：`deep_residual`
- 输入尺寸：`320 x 320`
- 输出网格：`10 x 10`
- 三框检测头：`num_boxes = 3`
- `head_type = "decoupled"`
- `assignment_strategy = "dynamic_cost"`
- `dynamic_topk = 2`
- anchors 与 box 参数化

本轮唯一主变量是分类损失：

- `cls_loss_mode = "varifocal"`
- `varifocal_alpha = 0.75`
- `varifocal_gamma = 2.0`

## 3. 对应实验

- 实验名：
  - `deep_residual_three_box_dynamicassign_varifocal`
- run_id：
  - `deep_residual_three_box_dynamicassign_varifocal_20260424_110543`
- 配置：
  - [deep_residual_three_box_dynamicassign_varifocal.toml](/home/lidz/YOLO/yolov0/configs/deep_residual_three_box_dynamicassign_varifocal.toml:1)
- 结果：
  - [result.txt](/home/lidz/YOLO/yolov0/logs/records/deep_residual_three_box_dynamicassign_varifocal_20260424_110543/result.txt:1)
- 元数据：
  - [metadata.json](/home/lidz/YOLO/yolov0/logs/records/deep_residual_three_box_dynamicassign_varifocal_20260424_110543/metadata.json:1)
- 评估：
  - [internal eval](/home/lidz/YOLO/yolov0/outputs/evaluations/deep_residual_three_box_dynamicassign_varifocal_eval.json:1)
  - [COCO subset eval](/home/lidz/YOLO/yolov0/outputs/evaluations/deep_residual_three_box_dynamicassign_varifocal_coco_eval.json:1)

## 4. 结果

- `best val total = 2.278960`
- `best epoch = 20`
- internal `mAP@0.5 = 0.079361`
- internal `precision = 0.045218`
- internal `recall = 0.302465`
- internal `num_predictions = 351632`
- COCO `AP50 = 0.000814`
- COCO `AR@100 = 0.002625`

## 5. 结论

Round 6B 的训练曲线继续改善，最终验证总损失也优于 `Round 5C`，但更正式的检测指标没有同步提升：

- internal `mAP@0.5` 低于 `Round 5C`
- `precision` 低于 `Round 5C`
- COCO `AP50` 明显低于 `Round 5C` 和 `Round 6A`
- `num_predictions` 反而高于 `Round 5C`

因此当前可以得出结论：

- `varifocal-style classification` 在本轮设定下改善了训练目标值
- 但没有转化为更好的最终排序质量
- `Round 6B` 不应取代 `Round 5C` 作为当前训练侧主基线

当前更合理的保留方式是：

- 训练侧主基线：`Round 5C`
- 推理侧排序参考：`Round 6A`
- `Round 6B` 记录为一次有效但未胜出的训练端尝试
