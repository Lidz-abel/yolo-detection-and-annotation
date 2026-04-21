# Stage C Round 6A Change Notes

本轮只改推理端排序公式，不重训模型。

## 变更目标

- 固定 `Round 5C` 的 best checkpoint
- 不改 backbone、head、loss、assignment
- 只测试更强调 `objectness` 的评分是否能改善排序质量

## 核心改动

- 在 `utils/prediction.py` 中新增：
  - `score_alpha`
  - `score_beta`
- 新评分公式：
  - `score = obj^alpha * cls^beta`
- 本轮配置为：
  - `score_alpha = 2.0`
  - `score_beta = 1.0`

## 预期

- 减少低质量框通过排序进入最终结果
- 优先改善：
  - `precision`
  - `AP50`
- 同时尽量保持：
  - `recall`
  - `AR@100`
