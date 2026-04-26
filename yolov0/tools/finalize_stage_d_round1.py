#!/usr/bin/env python3
"""Finalize one Stage D formal run after the long-running training ends."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path


ROOT = Path("/home/lidz/YOLO/yolov0")
PYTHON = "/home/lidz/anaconda3/envs/yolov1/bin/python"


def parse_args() -> argparse.Namespace:
    """Parse the Stage D finalize arguments."""
    parser = argparse.ArgumentParser(description="Finalize one Stage D formal run.")
    parser.add_argument("--run-id", required=True, help="Completed training run id.")
    parser.add_argument("--config-name", required=True, help="Config filename in configs/.")
    parser.add_argument("--experiment-name", required=True, help="Stable experiment prefix used for eval json names.")
    return parser.parse_args()


def run_cmd(cmd: list[str]) -> None:
    """Run one subprocess from the project root and fail fast on errors."""
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def read_text(path: Path) -> str:
    """Read one UTF-8 text file."""
    return path.read_text(encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    """Write one UTF-8 text file."""
    path.write_text(text, encoding="utf-8")


def wait_for_completion(result_path: Path, metadata_path: Path) -> None:
    """Wait until train.py marks the formal run as completed."""
    while True:
        if result_path.exists() and "status = completed" in read_text(result_path):
            return
        if metadata_path.exists():
            metadata = json.loads(read_text(metadata_path))
            if metadata.get("status") == "completed":
                return
        time.sleep(60)


def parse_result_metrics(result_text: str) -> dict[str, float | int | str]:
    """Extract the main scalar values we need from result.txt."""
    patterns = {
        "best_epoch": r"best_epoch = (\d+)",
        "best_val_loss": r"best_val_loss = ([0-9.eE+-]+)",
        "parameter_total": r"parameter_total = (\d+)",
        "model_output_shape": r"model_output_shape = (.+)",
        "output_dir": r"output_dir = (.+)",
        "visualization_dir": r"visualization_dir = (.+)",
    }
    parsed: dict[str, float | int | str] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, result_text)
        if not match:
            continue
        value = match.group(1).strip()
        if key in {"best_epoch", "parameter_total"}:
            parsed[key] = int(value)
        elif key == "best_val_loss":
            parsed[key] = float(value)
        else:
            parsed[key] = value

    train_matches = re.findall(
        r"epoch = (\d+) \| total = ([0-9.]+) \| box = ([0-9.]+) \| obj = ([0-9.]+) \| cls = ([0-9.]+)",
        result_text,
    )
    val_section = result_text.split("val history:\n", 1)
    val_matches = []
    if len(val_section) == 2:
        val_matches = re.findall(
            r"epoch = (\d+) \| total = ([0-9.]+) \| box = ([0-9.]+) \| obj = ([0-9.]+) \| cls = ([0-9.]+)",
            val_section[1],
        )
    if train_matches:
        epoch, total, box, obj, cls = train_matches[-1]
        parsed.update(
            {
                "final_train_epoch": int(epoch),
                "final_train_total": float(total),
                "final_train_box": float(box),
                "final_train_obj": float(obj),
                "final_train_cls": float(cls),
            }
        )
    if val_matches:
        epoch, total, box, obj, cls = val_matches[-1]
        parsed.update(
            {
                "final_val_epoch": int(epoch),
                "final_val_total": float(total),
                "final_val_box": float(box),
                "final_val_obj": float(obj),
                "final_val_cls": float(cls),
            }
        )
    return parsed


def append_unique_row(path: Path, anchor_row: str, new_row: str) -> None:
    """Append one markdown row after one known row if it is missing."""
    text = read_text(path)
    if new_row in text:
        return
    if anchor_row in text:
        text = text.replace(anchor_row, anchor_row + "\n" + new_row, 1)
    else:
        text = text.rstrip() + "\n" + new_row + "\n"
    write_text(path, text)


def replace_checklist_item(path: Path, old: str, new: str) -> None:
    """Replace one checklist line with an updated state."""
    text = read_text(path)
    if old in text:
        text = text.replace(old, new)
        write_text(path, text)


def build_summary(
    run_id: str,
    config_name: str,
    experiment_name: str,
    result_metrics: dict[str, float | int | str],
    internal_eval: dict,
    coco_eval: dict,
) -> str:
    """Render the dedicated Stage D Round 1 markdown summary."""
    return f"""# Stage D Round 1 Summary

本文件总结 checkpoint 6 进入 Stage D 之后的第一轮正式实验：

- `deep_residual_multiscale_singlebox`

## 1. 实验目的

在单尺度三框已经完成多轮优化后，本轮不再继续局部调参，而是验证最小多尺度检测器是否能突破：

- `10x10` 单尺度的物理分辨率瓶颈
- 小目标与密集目标在同一尺度中的拥挤问题
- 单尺度三框不断出现的排序压力

## 2. 控制变量

以下内容尽量保持和 Stage C 成熟版本一致：

- backbone 仍然基于 `deep_residual`
- 输入尺寸仍然是 `320 x 320`
- 解耦检测头保留
- 完整 YOLO-style loss 保留
- 质量感知分类保留
- 评估协议保持不变

本轮主要新变量是：

- 双尺度输出：
  - `P4: 20x20`
  - `P5: 10x10`
- 每个尺度先使用 `1` 个框槽位

## 3. 对应实验

- 实验名：
  - `{experiment_name}`
- run_id：
  - `{run_id}`
- 配置：
  - [{config_name}](/home/lidz/YOLO/yolov0/configs/{config_name}:1)
- 结果：
  - [result.txt](/home/lidz/YOLO/yolov0/logs/records/{run_id}/result.txt:1)
- 元数据：
  - [metadata.json](/home/lidz/YOLO/yolov0/logs/records/{run_id}/metadata.json:1)
- 评估：
  - [internal eval](/home/lidz/YOLO/yolov0/outputs/evaluations/{experiment_name}_eval.json:1)
  - [COCO subset eval](/home/lidz/YOLO/yolov0/outputs/evaluations/{experiment_name}_coco_eval.json:1)

## 4. 训练结果

- 输出形状：
  - `{result_metrics.get("model_output_shape", "unknown")}`
- 参数量：
  - `{int(result_metrics.get("parameter_total", 0)) / 1_000_000:.2f}M`
- 最优 epoch：
  - `{result_metrics.get("best_epoch", "unknown")}`
- 最优验证总损失：
  - `{result_metrics.get("best_val_loss", float("nan")):.6f}`
- 最终训练总损失：
  - `{result_metrics.get("final_train_total", float("nan")):.6f}`
- 最终验证总损失：
  - `{result_metrics.get("final_val_total", float("nan")):.6f}`

## 5. 检测结果

- internal `mAP@0.5 = {internal_eval['map50']:.6f}`
- internal `precision = {internal_eval['precision']:.6f}`
- internal `recall = {internal_eval['recall']:.6f}`
- internal `num_predictions = {int(internal_eval['num_predictions'])}`
- COCO `AP50 = {coco_eval['coco_ap50']:.6f}`
- COCO `AR@100 = {coco_eval['coco_ar100']:.6f}`

## 6. 初步结论

Stage D Round 1 用最小双尺度单框方案回答了一个核心问题：

- 多尺度本身是否比单尺度三框更有效？

后续判断将重点参考：

- 是否明显改善 AP50
- 是否明显改善 AR@100
- 可视化中小目标与密集目标是否更清晰
- 是否降低单尺度阶段长期存在的过密候选框问题
"""


def finalize(args: argparse.Namespace) -> None:
    """Wait for the run, evaluate it, summarize it, and commit it."""
    run_id = args.run_id
    config_name = args.config_name
    experiment_name = args.experiment_name
    record_dir = ROOT / "logs" / "records" / run_id
    result_path = record_dir / "result.txt"
    metadata_path = record_dir / "metadata.json"
    wait_for_completion(result_path, metadata_path)

    checkpoint = ROOT / "outputs" / run_id / "best.pth"
    config_path = ROOT / "configs" / config_name
    eval_json = ROOT / "outputs" / "evaluations" / f"{experiment_name}_eval.json"
    coco_json = ROOT / "outputs" / "evaluations" / f"{experiment_name}_coco_eval.json"

    run_cmd(
        [
            PYTHON,
            "tools/evaluate.py",
            "--config",
            str(config_path),
            "--checkpoint",
            str(checkpoint),
            "--score-threshold",
            "0.05",
            "--score-alpha",
            "2.0",
            "--score-beta",
            "1.0",
            "--top-k",
            "100",
            "--nms-iou-threshold",
            "0.5",
            "--map-iou-threshold",
            "0.5",
            "--output-json",
            str(eval_json),
        ]
    )
    run_cmd(
        [
            PYTHON,
            "tools/evaluate_coco.py",
            "--config",
            str(config_path),
            "--checkpoint",
            str(checkpoint),
            "--score-threshold",
            "0.05",
            "--score-alpha",
            "2.0",
            "--score-beta",
            "1.0",
            "--top-k",
            "100",
            "--nms-iou-threshold",
            "0.5",
            "--output-json",
            str(coco_json),
        ]
    )

    result_metrics = parse_result_metrics(read_text(result_path))
    internal_eval = json.loads(read_text(eval_json))
    coco_eval = json.loads(read_text(coco_json))

    summary_path = ROOT / "logs" / "stage_d_round1_summary.md"
    write_text(summary_path, build_summary(run_id, config_name, experiment_name, result_metrics, internal_eval, coco_eval))

    index_path = ROOT / "logs" / "experiment_result_index.md"
    anchor_row = "| deep_residual_three_box_dynamicassign_varifocal | `deep_residual_three_box_dynamicassign_varifocal_20260424_110543` | [deep_residual_three_box_dynamicassign_varifocal.toml](/home/lidz/YOLO/yolov0/configs/deep_residual_three_box_dynamicassign_varifocal.toml:1) | [result.txt](/home/lidz/YOLO/yolov0/logs/records/deep_residual_three_box_dynamicassign_varifocal_20260424_110543/result.txt:1) | [metadata.json](/home/lidz/YOLO/yolov0/logs/records/deep_residual_three_box_dynamicassign_varifocal_20260424_110543/metadata.json:1) | [eval.json](/home/lidz/YOLO/yolov0/outputs/evaluations/deep_residual_three_box_dynamicassign_varifocal_eval.json:1), [coco_eval.json](/home/lidz/YOLO/yolov0/outputs/evaluations/deep_residual_three_box_dynamicassign_varifocal_coco_eval.json:1) | [outputs](/home/lidz/YOLO/yolov0/outputs/deep_residual_three_box_dynamicassign_varifocal_20260424_110543:1) |"
    new_row = f"| {experiment_name} | `{run_id}` | [{config_name}](/home/lidz/YOLO/yolov0/configs/{config_name}:1) | [result.txt](/home/lidz/YOLO/yolov0/logs/records/{run_id}/result.txt:1) | [metadata.json](/home/lidz/YOLO/yolov0/logs/records/{run_id}/metadata.json:1) | [eval.json](/home/lidz/YOLO/yolov0/outputs/evaluations/{experiment_name}_eval.json:1), [coco_eval.json](/home/lidz/YOLO/yolov0/outputs/evaluations/{experiment_name}_coco_eval.json:1) | [outputs](/home/lidz/YOLO/yolov0/outputs/{run_id}:1) |"
    append_unique_row(index_path, anchor_row, new_row)

    eval_summary_path = ROOT / "logs" / "detection_evaluation_summary.md"
    eval_text = read_text(eval_summary_path).rstrip()
    addition = f"""

### 5.9 Stage D Round 1 introduces the first multiscale detector

The new multiscale Stage-D branch keeps the strong `deep_residual` backbone family,
but replaces the single `10x10` output with two scales:

- `P4: 20x20`
- `P5: 10x10`

This first round keeps the design intentionally minimal:

- one box slot per scale
- decoupled heads
- the current full YOLO-style loss family
- no heavy neck or extra augmentation

The formal run is:

- `{run_id}`

Key results:

- internal `mAP@0.5 = {internal_eval['map50']:.6f}`
- internal `precision = {internal_eval['precision']:.6f}`
- internal `recall = {internal_eval['recall']:.6f}`
- COCO `AP50 = {coco_eval['coco_ap50']:.6f}`
- COCO `AR@100 = {coco_eval['coco_ar100']:.6f}`

This run is the first Stage-D reference point and should be compared against:

- `Round 5C` as the strongest training-side single-scale baseline
- `Round 6A` as the strongest ranking-side single-scale reference
"""
    if "### 5.9 Stage D Round 1 introduces the first multiscale detector" not in eval_text:
        write_text(eval_summary_path, eval_text + addition + "\n")

    checklist_path = ROOT / "logs" / "stage_d_execution_checklist.md"
    replace_checklist_item(checklist_path, "- [ ] 启动 `deep_residual_multiscale_singlebox` full-run", "- [x] 启动 `deep_residual_multiscale_singlebox` full-run")
    replace_checklist_item(checklist_path, "- [ ] 生成 `config.toml / metadata.json / result.txt`", "- [x] 生成 `config.toml / metadata.json / result.txt`")
    replace_checklist_item(checklist_path, "- [ ] internal eval", "- [x] internal eval")
    replace_checklist_item(checklist_path, "- [ ] COCO subset eval", "- [x] COCO subset eval")
    replace_checklist_item(checklist_path, "- [ ] GT vs Pred 可视化", "- [x] GT vs Pred 可视化")
    replace_checklist_item(checklist_path, "- [ ] 阶段总结", "- [x] 阶段总结")
    replace_checklist_item(checklist_path, "- [ ] 更新实验索引", "- [x] 更新实验索引")

    run_cmd(["git", "add", "configs/deep_residual_multiscale_singlebox.toml"])
    run_cmd(["git", "add", "logs/stage_d_plan.md"])
    run_cmd(["git", "add", "logs/stage_d_execution_checklist.md"])
    run_cmd(["git", "add", "logs/stage_d_round1_change_notes.md"])
    run_cmd(["git", "add", "logs/stage_d_round1_summary.md"])
    run_cmd(["git", "add", "logs/experiment_result_index.md"])
    run_cmd(["git", "add", "logs/detection_evaluation_summary.md"])
    run_cmd(["git", "add", "logs/records/" + run_id])
    run_cmd(["git", "add", "models/backbone.py"])
    run_cmd(["git", "add", "models/detector.py"])
    run_cmd(["git", "add", "data/detection_dataset.py"])
    run_cmd(["git", "add", "losses/yolo_loss.py"])
    run_cmd(["git", "add", "utils/config.py"])
    run_cmd(["git", "add", "utils/modeling.py"])
    run_cmd(["git", "add", "utils/prediction.py"])
    run_cmd(["git", "add", "utils/evaluation.py"])
    run_cmd(["git", "add", "utils/coco_eval.py"])
    run_cmd(["git", "add", "utils/visualization.py"])
    run_cmd(["git", "add", "engine/trainer.py"])
    run_cmd(["git", "add", "tools/train.py"])
    run_cmd(["git", "add", "tools/evaluate.py"])
    run_cmd(["git", "add", "tools/evaluate_coco.py"])
    run_cmd(["git", "add", "tools/finalize_stage_d_round1.py"])
    run_cmd(["git", "add", "outputs/evaluations/" + f"{experiment_name}_eval.json"])
    run_cmd(["git", "add", "outputs/evaluations/" + f"{experiment_name}_coco_eval.json"])
    run_cmd(["git", "commit", "-m", "Record Stage D Round 1 multiscale single-box results"])


if __name__ == "__main__":
    finalize(parse_args())
