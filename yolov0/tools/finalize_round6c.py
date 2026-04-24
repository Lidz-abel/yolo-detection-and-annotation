#!/usr/bin/env python3
"""Finalize the long-running Round 6C experiment after training completes.

Round 6C keeps the Round 5C detector design but tightens dynamic assignment by
reducing `dynamic_topk`. This helper waits for the full-run experiment to end,
then evaluates, summarizes, and updates the experiment index.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path("/home/lidz/YOLO/yolov0")
PYTHON = "/home/lidz/anaconda3/envs/yolov1/bin/python"


def run_cmd(cmd: list[str]) -> None:
    """Run one subprocess from the project root and fail fast on errors."""
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def read_text(path: Path) -> str:
    """Read a UTF-8 text file."""
    return path.read_text(encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    """Write a UTF-8 text file."""
    path.write_text(text, encoding="utf-8")


def process_exists(pid: int) -> bool:
    """Return whether one Linux PID still exists."""
    return Path(f"/proc/{pid}").exists()


def wait_for_completion(result_path: Path, metadata_path: Path, train_pid: int) -> None:
    """Block until the training run is fully marked completed."""
    while True:
        if result_path.exists() and "status = completed" in read_text(result_path):
            return
        if metadata_path.exists():
            metadata = json.loads(read_text(metadata_path))
            if metadata.get("status") == "completed":
                return
        if train_pid and not process_exists(train_pid):
            time.sleep(30)
        time.sleep(60)


def parse_result_metrics(result_text: str) -> dict[str, float | int | str]:
    """Extract the main scalar fields we need from one result.txt file."""
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


def append_unique_line(path: Path, marker: str, line: str) -> None:
    """Insert one markdown line before a marker if it is not already present."""
    text = read_text(path)
    if line in text:
        return
    if marker in text:
        text = text.replace(marker, line + "\n" + marker, 1)
    else:
        text = text.rstrip() + "\n" + line + "\n"
    write_text(path, text)


def append_unique_row(path: Path, anchor_row: str, new_row: str) -> None:
    """Append one markdown row after a known row if it is missing."""
    text = read_text(path)
    if new_row in text:
        return
    if anchor_row in text:
        text = text.replace(anchor_row, anchor_row + "\n" + new_row, 1)
    else:
        text = text.rstrip() + "\n" + new_row + "\n"
    write_text(path, text)


def build_summary(run_id: str, result_metrics: dict[str, float | int | str], internal_eval: dict, coco_eval: dict) -> str:
    """Render the dedicated Round 6C markdown summary."""
    return f"""# Stage C Round 6C Summary

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
  - `{run_id}`
- 配置：
  - [deep_residual_three_box_dynamicassign_topk1.toml](/home/lidz/YOLO/yolov0/configs/deep_residual_three_box_dynamicassign_topk1.toml:1)
- 结果：
  - [result.txt](/home/lidz/YOLO/yolov0/logs/records/{run_id}/result.txt:1)
- 元数据：
  - [metadata.json](/home/lidz/YOLO/yolov0/logs/records/{run_id}/metadata.json:1)
- 评估：
  - [internal eval](/home/lidz/YOLO/yolov0/outputs/evaluations/deep_residual_three_box_dynamicassign_topk1_eval.json:1)
  - [COCO subset eval](/home/lidz/YOLO/yolov0/outputs/evaluations/deep_residual_three_box_dynamicassign_topk1_coco_eval.json:1)

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

## 5. 与 Round 5C 对比

| 指标 | Round 5C | Round 6C | 变化解读 |
| --- | ---: | ---: | --- |
| Best Val Total | `2.290735` | `{result_metrics.get("best_val_loss", float("nan")):.6f}` | 待本轮结论解释 |
| Internal mAP@0.5 | `0.086128` | `{internal_eval['map50']:.6f}` | 待本轮结论解释 |
| Internal Precision | `0.048140` | `{internal_eval['precision']:.6f}` | 待本轮结论解释 |
| Internal Recall | `0.306746` | `{internal_eval['recall']:.6f}` | 待本轮结论解释 |
| Num Predictions | `334960` | `{int(internal_eval['num_predictions'])}` | 候选框数量变化 |
| COCO AP50 | `0.001150` | `{coco_eval['coco_ap50']:.6f}` | 待本轮结论解释 |
| COCO AR@100 | `0.003097` | `{coco_eval['coco_ar100']:.6f}` | 待本轮结论解释 |

## 6. 主要结论

- internal `mAP@0.5 = {internal_eval['map50']:.6f}`
- internal `precision = {internal_eval['precision']:.6f}`
- internal `recall = {internal_eval['recall']:.6f}`
- COCO `AP50 = {coco_eval['coco_ap50']:.6f}`
- COCO `AR@100 = {coco_eval['coco_ar100']:.6f}`

Round 6C 的核心问题是：

- `dynamic_topk = 1` 是否能降低套娃框和多余预测；
- 同时尽量保住 Round 5C 已建立起来的召回优势。

如果这轮能提高 AP50 且不明显损伤 AR@100，那么后续就可以继续沿着动态分配收紧的方向做更轻量微调；否则应当保留 Round 5C 作为当前动态分配主基线。
"""


def finalize(args: argparse.Namespace) -> None:
    """Run the full Round 6C post-processing pipeline."""
    run_id = args.run_id
    record_dir = ROOT / "logs" / "records" / run_id
    result_path = record_dir / "result.txt"
    metadata_path = record_dir / "metadata.json"
    wait_for_completion(result_path, metadata_path, args.train_pid)

    checkpoint = ROOT / "outputs" / run_id / "best.pth"
    config_path = ROOT / "configs" / "deep_residual_three_box_dynamicassign_topk1.toml"
    eval_json = ROOT / "outputs" / "evaluations" / "deep_residual_three_box_dynamicassign_topk1_eval.json"
    coco_json = ROOT / "outputs" / "evaluations" / "deep_residual_three_box_dynamicassign_topk1_coco_eval.json"

    run_cmd(
        [
            PYTHON,
            "-u",
            "tools/evaluate.py",
            "--config",
            str(config_path),
            "--checkpoint",
            str(checkpoint),
            "--score-threshold",
            "0.05",
            "--score-alpha",
            "1.0",
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
            "-u",
            "tools/evaluate_coco.py",
            "--config",
            str(config_path),
            "--checkpoint",
            str(checkpoint),
            "--score-threshold",
            "0.05",
            "--score-alpha",
            "1.0",
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

    summary_path = ROOT / "logs" / "stagec_round6c_summary.md"
    write_text(summary_path, build_summary(run_id, result_metrics, internal_eval, coco_eval))

    index_path = ROOT / "logs" / "experiment_result_index.md"
    index_row = (
        f"| deep_residual_three_box_dynamicassign_topk1 | `{run_id}` | "
        f"[deep_residual_three_box_dynamicassign_topk1.toml](/home/lidz/YOLO/yolov0/configs/deep_residual_three_box_dynamicassign_topk1.toml:1) | "
        f"[result.txt](/home/lidz/YOLO/yolov0/logs/records/{run_id}/result.txt:1) | "
        f"[metadata.json](/home/lidz/YOLO/yolov0/logs/records/{run_id}/metadata.json:1) | "
        f"[eval.json](/home/lidz/YOLO/yolov0/outputs/evaluations/deep_residual_three_box_dynamicassign_topk1_eval.json:1), "
        f"[coco_eval.json](/home/lidz/YOLO/yolov0/outputs/evaluations/deep_residual_three_box_dynamicassign_topk1_coco_eval.json:1) | "
        f"[outputs](/home/lidz/YOLO/yolov0/outputs/{run_id}:1) |"
    )
    append_unique_line(index_path, "\n## 使用规则\n", index_row)

    summary_note_path = ROOT / "logs" / "detection_evaluation_summary.md"
    json_bullet = (
        "- `/home/lidz/YOLO/yolov0/outputs/evaluations/"
        "deep_residual_three_box_dynamicassign_topk1_coco_eval.json`"
    )
    append_unique_line(summary_note_path, "\n### 4.1 Result Table\n", json_bullet)
    anchor_row = (
        "| `deep_residual` + three-box + dynamic-assign | 3 | 28.92M | 42.66G | 98.41 | 0.000299 | 0.001150 | 0.000120 | 0.003097 |"
    )
    table_row = (
        "| `deep_residual` + three-box + dynamic-assign + topk1 | 3 | "
        f"{internal_eval['params_total'] / 1_000_000:.2f}M | {internal_eval['total_gflops']:.2f}G | {internal_eval['images_per_second']:.2f} | "
        f"{coco_eval['coco_ap']:.6f} | {coco_eval['coco_ap50']:.6f} | {coco_eval['coco_ap75']:.6f} | {coco_eval['coco_ar100']:.6f} |"
    )
    append_unique_row(summary_note_path, anchor_row, table_row)


def main() -> int:
    """Parse args and finalize Round 6C once the full-run is done."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--train-pid", type=int, required=True)
    args = parser.parse_args()
    finalize(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
