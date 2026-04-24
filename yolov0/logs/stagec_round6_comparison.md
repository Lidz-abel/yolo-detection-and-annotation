# Stage C Round 6 Comparison

本文件对比当前三组关键实验：

- `Round 5C`
- `Round 6A`
- `Round 6C`
- `Round 6B`

## 1. 关键指标对比

| 指标 | Round 5C | Round 6A | Round 6B | Round 6C |
| --- | ---: | ---: | ---: | ---: |
| Best Val Total | `2.290735` | `2.290735` | `2.278960` | `2.407872` |
| Internal mAP@0.5 | `0.086128` | `0.085815` | `0.079361` | `0.057019` |
| Internal Precision | `0.048140` | `0.084336` | `0.045218` | `0.031924` |
| Internal Recall | `0.306746` | `0.280456` | `0.302465` | `0.270526` |
| Num Predictions | `334960` | `174813` | `351632` | `445458` |
| COCO AP50 | `0.001150` | `0.001336` | `0.000814` | `0.000854` |
| COCO AR@100 | `0.003097` | `0.002482` | `0.002625` | `0.003093` |

## 2. 结论

### Round 5C

- 当前最强的训练侧三框基线
- 在召回和整体覆盖能力上最均衡
- 但预测框数量仍然偏多

### Round 6A

- 当前最强的推理侧排序方案
- 通过 `score = obj^2 * cls` 明显提升 precision 和 COCO AP50
- 代价是 recall 和 AR@100 小幅回落

### Round 6C

- `dynamic_topk = 1` 过度收紧动态分配
- precision、recall、mAP@0.5、COCO AP50 全面低于 Round 5C
- 预测框数量反而升到 `445458`
- 不适合作为下一轮训练基线

### Round 6B

- 训练端引入 `varifocal-style classification`
- best val total 降到 `2.278960`，是当前四组里最好的训练损失
- 但 internal `mAP@0.5`、`precision`、COCO `AP50` 都低于 Round 5C
- `num_predictions` 升到 `351632`
- 说明本轮改动改善了训练目标值，但没有带来更好的最终排序质量

## 3. 对 Round 6B 的影响

Round 6B 不应建立在 Round 6C 上，而应回退到：

- 训练侧基线：`Round 5C`
- 推理侧参考：`Round 6A`

也就是说，Round 6B 采用：

- `deep_residual`
- three-box
- decoupled head
- dynamic assignment (`dynamic_topk = 2`)
- 训练端引入更强的 quality-aware / varifocal-style classification

目标是：

- 保留 Round 5C 的召回与覆盖能力
- 同时在训练端提升排序质量
- 最终进一步接近或超过 Round 6A 的 AP50 优势

## 4. 当前阶段结论

在当前四组对比里：

- `Round 5C` 仍然是最强训练侧基线
- `Round 6A` 仍然是最强推理侧排序方案
- `Round 6B` 没有取代 `Round 5C`
- `Round 6C` 也不保留

所以当前最合理的工作基线仍然是：

- 训练侧：`Round 5C`
- 推理侧：`Round 6A`
