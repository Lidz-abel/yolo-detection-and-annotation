# Stage C Round 6A Summary

本文件总结 Stage C 第六轮改进中的第一步：

- `Round 6A: 固定 Round 5C 权重，只调整推理排序公式`

## 1. 实验目的

Round 5C 已经把三框版本推进到当前最强的召回与 AP50 水平，但也留下了：

- 预测框数量偏多
- 一物多框仍然存在
- 排序质量仍然弱于覆盖能力

因此 Round 6A 不再重训模型，而是固定：

- `deep_residual + three-box + quality-aware cls + decoupled head + dynamic assignment`

只改推理端评分公式，观察更强调 objectness 后：

- `AP50`
- `precision`
- 可视化质量

是否能改善。

## 2. 控制变量

以下内容全部保持和 Round 5C 一致：

- backbone：`deep_residual`
- 输入尺寸：`320 x 320`
- 输出网格：`10 x 10`
- 三框检测头：`num_boxes = 3`
- `head_type = "decoupled"`
- `assignment_strategy = "dynamic_cost"`
- anchors 与动态分配超参数
- source checkpoint：
  - `/home/lidz/YOLO/yolov0/outputs/deep_residual_three_box_dynamicassign_20260420_030501/best.pth`

本轮唯一主变量是评分公式：

- `score = obj^alpha * cls^beta`
- 其中：
  - `alpha = 2.0`
  - `beta = 1.0`

## 3. 对应实验

- 实验名：
  - `deep_residual_three_box_dynamicassign_scoretune`
- run_id：
  - `deep_residual_three_box_dynamicassign_scoretune_20260421_182204`
- 配置：
  - [deep_residual_three_box_dynamicassign_scoretune.toml](/home/lidz/YOLO/yolov0/configs/deep_residual_three_box_dynamicassign_scoretune.toml:1)
- 结果：
  - [result.txt](/home/lidz/YOLO/yolov0/logs/records/deep_residual_three_box_dynamicassign_scoretune_20260421_182204/result.txt:1)
- 元数据：
  - [metadata.json](/home/lidz/YOLO/yolov0/logs/records/deep_residual_three_box_dynamicassign_scoretune_20260421_182204/metadata.json:1)
- 评估：
  - [internal eval](/home/lidz/YOLO/yolov0/outputs/evaluations/deep_residual_three_box_dynamicassign_scoretune_eval.json:1)
  - [COCO subset eval](/home/lidz/YOLO/yolov0/outputs/evaluations/deep_residual_three_box_dynamicassign_scoretune_coco_eval.json:1)

## 4. 结果

- internal `mAP@0.5 = 0.085815`
- internal `precision = 0.084336`
- internal `recall = 0.280456`
- internal `num_predictions = 174813`
- COCO `AP50 = 0.001336`
- COCO `AR@100 = 0.002482`

可视化使用：

- `score_threshold = 0.25`
- `score_alpha = 2.0`
- `score_beta = 1.0`

## 5. 结论

Round 6A 的意义不在于扩大召回，而在于验证：

- 当模型已经学会“找到更多候选框”后
- 更强调 objectness 的排序公式
- 是否能让高质量框自然排到更前面

这轮结果应当与 Round 5C 做直接对比，再决定下一步继续改训练端还是先改动态分配超参数。
