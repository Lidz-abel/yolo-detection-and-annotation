# Stage C Round-3A Anchor Review

本文件记录 Stage C Round-3A 的 anchor 重估结论。

## 目的

在继续修改三框机制之前，先验证当前 3 个 anchor 是否已经与训练集分布匹配。
如果当前 anchor 明显不适配，则应当优先重估 anchor；如果重估结果与当前配置一致，
则说明下一步不必把精力放在 anchor 数值本身。

## 使用脚本

- [fit_anchors.py](/home/lidz/YOLO/yolov0/tools/fit_anchors.py:1)

输入数据：

- `/home/lidz/YOLO/DataSet/Unified/manifests/all_train.jsonl`

聚类设定：

- `num_anchors = 3`
- 距离度量：`1 - IoU(width,height)`
- 输出：归一化 anchor 宽高

## 结果

重新拟合得到的 3 个 anchor 为：

- `0.052141,0.076305`
- `0.199926,0.288289`
- `0.609263,0.649287`

它们与当前 Stage C 配置中的 anchor **完全一致**。

## 覆盖统计

对全训练集所有 GT 宽高做 best-anchor 覆盖统计，得到：

- `mean best IoU ≈ 0.4357`
- `p10 ≈ 0.1747`
- `p25 ≈ 0.3061`
- `p50 ≈ 0.4250`
- `p75 ≈ 0.5651`
- `p90 ≈ 0.7011`

## 结论

Stage C 当前使用的 3 个 anchor 已经是数据驱动重估后的结果，
因此 Round-3A **没有产生新的 anchor 变量**。

这意味着：

1. 不需要为了“重新拟合 anchor”重复运行一次和 Round-2 完全等价的 full experiment。
2. 下一步更值得继续改的方向不是 anchor 数值本身，而是：
   - soft objectness 的冷启动问题
   - 三框的匹配与分配规则

## 当前决定

Round-3A 记录完成，但不单独生成新的 full-run。

后续实验直接进入：

- Stage C Round-3B：给 soft objectness target 引入下限。
