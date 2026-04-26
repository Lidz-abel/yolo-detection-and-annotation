# yolo_final

`yolo_final` 是 checkpoint 6 后续阶段的新正式工程目录，目标是从 `yolov0` 的单尺度实验线过渡到一个更接近主流 YOLO 的**双尺度三框检测器**。

本目录不再以教学式逐步试探为主，而是以**工程化双尺度三框方案**为主线推进：

- 双尺度输出
- 每尺度三框
- 解耦检测头
- 完整 YOLO-style loss
- 动态分配
- 正式实验记录

## 当前目标

当前已经从“正式方案定义”推进到了“正式实现接线完成”阶段：

- 工程目录结构已建立
- 双尺度三框架构蓝图已落盘
- 正式实验计划已落盘
- 双尺度三框正式配置已建立
- 双尺度三框主链路已经接通并通过 smoke 检查

后续的工作重点将转入：

- 第一轮双尺度三框正式 full-run
- internal eval / COCO subset eval
- 可视化与对比总结

## 目录结构

```text
yolo_final/
├── README.md
├── configs/
├── data/
├── engine/
├── logs/
├── losses/
├── models/
├── outputs/
├── report/
├── runs/
├── tools/
└── utils/
```

## 当前文档

- 正式执行计划：
  - [dual_scale_three_box_master_plan.md](/home/lidz/YOLO/yolo_final/logs/dual_scale_three_box_master_plan.md:1)
- 架构蓝图：
  - [dual_scale_three_box_architecture.md](/home/lidz/YOLO/yolo_final/logs/dual_scale_three_box_architecture.md:1)
- 实验协议：
  - [experiment_protocol.md](/home/lidz/YOLO/yolo_final/logs/experiment_protocol.md:1)
- 正式配置模板：
  - [dual_scale_three_box_formal.toml](/home/lidz/YOLO/yolo_final/configs/dual_scale_three_box_formal.toml:1)
- 双尺度三框 smoke 检查：
  - [smoke_dual_scale_three_box.py](/home/lidz/YOLO/yolo_final/tools/smoke_dual_scale_three_box.py:1)
- 双尺度三框实现说明：
  - [dual_scale_three_box_implementation_notes.md](/home/lidz/YOLO/yolo_final/logs/dual_scale_three_box_implementation_notes.md:1)
- 当前可运行基线配置：
  - [multiscale_singlebox_baseline.toml](/home/lidz/YOLO/yolo_final/configs/multiscale_singlebox_baseline.toml:1)
- 实现状态说明：
  - [implementation_status.md](/home/lidz/YOLO/yolo_final/logs/implementation_status.md:1)

## 当前代码状态

`yolo_final` 现在已经把 `yolov0` 中成熟的训练、评估、可视化与完整 loss 链路迁移过来了。

当前可以这样理解：

- **已经实现并可作为工程基线使用的部分**
  - 单尺度训练链路
  - 单尺度三框链路
  - 最小双尺度单框链路
  - 双尺度三框正式实现接线
  - 每尺度独立 anchor 解析与使用
  - 双尺度三框 prediction / eval / COCO eval / visualization 路径

- **目前仍未完成正式实验验证的部分**
  - 双尺度三框 full-run
  - 双尺度三框正式指标与可视化归档
  - 双尺度三框后续优化版本
