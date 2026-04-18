# MiniYOLO

## 目录结构

```text
miniyolo/
├── configs/
├── data/
├── models/
├── losses/
├── engine/
├── utils/
├── tools/
├── runs/
├── outputs/
├── logs/
└── report/
```

各目录职责如下：

- `configs/`
  - 保存训练和模型配置。
  - 例如输入尺寸、网格大小、类别数、学习率、batch size 等。

- `data/`
  - 保存 `Dataset`、标注编码、训练样本读取逻辑。
  - 后续会把统一后的 manifest 数据接入这里。

- `models/`
  - 保存模型结构。
  - 包括极简 CNN backbone、检测 head、完整 MiniYOLO 模型封装。

- `losses/`
  - 保存极简 YOLOv1 风格损失函数。
  - 重点包含格子分类损失和 bbox 中心点、宽高回归损失。

- `engine/`
  - 保存训练与验证过程。
  - 例如单轮训练、loss 统计、优化器 step 等。

- `utils/`
  - 保存通用工具函数。
  - 例如 tensor shape 检查、可视化、日志辅助函数。

- `tools/`
  - 保存直接运行的脚本入口。
  - 例如模型结构检查、训练脚本、预测可视化脚本。

- `runs/`
  - 保存 TensorBoard 运行日志。

- `outputs/`
  - 保存预测可视化图片、模型中间输出、训练生成的结果文件。

- `logs/`
  - 保存实验记录和实现说明，方便后续撰写报告。

- `report/`
  - 保存检查点 4 的 LaTeX 报告和最终 PDF。

## 开发顺序

建议按下面顺序推进：

1. 在 `models/` 中实现极简 CNN backbone。
2. 在 `tools/` 中增加模型结构检查脚本，输出每层 tensor shape。
3. 在 `losses/` 中实现简化版 YOLO loss。
4. 在 `data/` 中准备最小训练用 `Dataset` 和 target 编码方式。
5. 在 `engine/` 和 `tools/` 中跑通训练循环。
6. 在 `runs/` 中记录 TensorBoard loss 曲线。
7. 在 `outputs/` 中保存预测框与真实框的可视化结果。
8. 在 `report/` 中整理最终实验报告。

## 当前约定

- 工程重点是“跑通最小检测链路”，不是追求检测精度。
- 每一步都应尽量保留 shape 信息和实验记录，便于报告撰写。
- 后续的实现优先保证：
  - 前向 shape 正确
  - loss 可计算
  - backward 可运行
  - training loss 可下降

## 后续重点文件

后续预计优先创建以下文件：

- `models/minimal_backbone.py`
- `models/detection_head.py`
- `models/miniyolo.py`
- `losses/minimal_yolo_loss.py`
- `data/detection_dataset.py`
- `data/target_encoder.py`
- `tools/inspect_model.py`
- `tools/train.py`
- `tools/visualize_predictions.py`

## 输出目标

检查点 4 最终需要形成以下产物：

- 每层 tensor shape 说明
- training loss 下降曲线
- 预测框与真实框对比图
- 检查点 4 PDF 报告
