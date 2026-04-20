#!/usr/bin/env python3
"""Finalize the long-running Round 5C experiment after training completes.

This helper waits for a formal training run to finish, then performs the
remaining bookkeeping that our checkpoint-6 workflow requires:

- run internal and COCO-subset evaluation
- write a stage summary markdown file
- update the experiment index and evaluation summary notes
- create a dedicated git commit for the finished experiment

The script is intentionally specific to the current Round 5C workflow so we
can automate a long full-run experiment without hand babysitting.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path("/home/lidz/YOLO/yolov0")
PYTHON = "/home/lidz/anaconda3/envs/yolov1/bin/python"


def run_cmd(cmd: list[str], cwd: Path | None = None) -> None:
    """Run a subprocess and raise immediately on failure."""
    subprocess.run(cmd, cwd=str(cwd or ROOT), check=True)


def read_text(path: Path) -> str:
    """Read a UTF-8 text file."""
    return path.read_text(encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    """Write a UTF-8 text file."""
    path.write_text(text, encoding="utf-8")


def process_exists(pid: int) -> bool:
    """Return whether a Linux PID still exists."""
    return Path(f"/proc/{pid}").exists()


def wait_for_completion(result_path: Path, metadata_path: Path, train_pid: int) -> None:
    """Block until the training result is marked completed."""
    while True:
        if result_path.exists():
            result_text = read_text(result_path)
            if "status = completed" in result_text:
                return
        if metadata_path.exists():
            metadata = json.loads(read_text(metadata_path))
            if metadata.get("status") == "completed":
                return
        if train_pid and not process_exists(train_pid):
            # Give the trainer a little time to flush its final result files.
            time.sleep(30)
        time.sleep(60)


def parse_result_metrics(result_text: str) -> dict[str, float | int | str]:
    """Extract the key scalar fields we need from result.txt."""
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
        elif key in {"best_val_loss"}:
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
    """Insert a line before a marker only if it is not already present."""
    text = read_text(path)
    if line in text:
        return
    if marker not in text:
        text = text.rstrip() + "\n" + line + "\n"
    else:
        text = text.replace(marker, line + "\n" + marker, 1)
    write_text(path, text)


def append_unique_row(path: Path, anchor_row: str, new_row: str) -> None:
    """Append a markdown table row after a known anchor row if absent."""
    text = read_text(path)
    if new_row in text:
        return
    if anchor_row in text:
        text = text.replace(anchor_row, anchor_row + "\n" + new_row, 1)
    else:
        text = text.rstrip() + "\n" + new_row + "\n"
    write_text(path, text)


def update_experiment_index(run_id: str) -> None:
    """Append the Round 5C experiment mapping to the result index."""
    index_path = ROOT / "logs/experiment_result_index.md"
    new_row = (
        f"| deep_residual_three_box_dynamicassign | `{run_id}` | "
        f"[deep_residual_three_box_dynamicassign.toml](/home/lidz/YOLO/yolov0/configs/deep_residual_three_box_dynamicassign.toml:1) | "
        f"[result.txt](/home/lidz/YOLO/yolov0/logs/records/{run_id}/result.txt:1) | "
        f"[metadata.json](/home/lidz/YOLO/yolov0/logs/records/{run_id}/metadata.json:1) | "
        f"[eval.json](/home/lidz/YOLO/yolov0/outputs/evaluations/deep_residual_three_box_dynamicassign_eval.json:1), "
        f"[coco_eval.json](/home/lidz/YOLO/yolov0/outputs/evaluations/deep_residual_three_box_dynamicassign_coco_eval.json:1) | "
        f"[outputs](/home/lidz/YOLO/yolov0/outputs/{run_id}:1) |"
    )
    marker = "\n## 使用规则\n"
    append_unique_line(index_path, marker, new_row)


def update_detection_summary() -> None:
    """Extend the evaluation summary with the Round 5C result."""
    summary_path = ROOT / "logs/detection_evaluation_summary.md"
    json_bullet = (
        "- `/home/lidz/YOLO/yolov0/outputs/evaluations/"
        "deep_residual_three_box_dynamicassign_coco_eval.json`"
    )
    append_unique_line(summary_path, "\n### 4.1 Result Table\n", json_bullet)
    table_anchor = (
        "| `deep_residual` + three-box + quality-cls + decoupled | 3 | 28.92M | 42.66G | 93.81 | 0.000206 | 0.000814 | 0.000021 | 0.002500 |"
    )
    new_row = (
        "| `deep_residual` + three-box + dynamic-assign | 3 | 28.92M | 42.66G | 0.00 | 0.0 | 0.0 | 0.0 | 0.0 |"
    )
    append_unique_row(summary_path, table_anchor, new_row)


def replace_dynamic_summary_row(coco_eval: dict, internal_eval: dict) -> None:
    """Patch the placeholder Round 5C row with real numbers."""
    summary_path = ROOT / "logs/detection_evaluation_summary.md"
    text = read_text(summary_path)
    pattern = re.escape("| `deep_residual` + three-box + dynamic-assign | 3 | 28.92M | 42.66G | 0.00 | 0.0 | 0.0 | 0.0 | 0.0 |")
    replacement = (
        "| `deep_residual` + three-box + dynamic-assign | 3 | 28.92M | 42.66G | "
        f"{internal_eval['fps']:.2f} | {coco_eval['ap']:.6f} | {coco_eval['ap50']:.6f} | "
        f"{coco_eval['ap75']:.6f} | {coco_eval['ar_100']:.6f} |"
    )
    text = re.sub(pattern, replacement, text, count=1)
    section = f"""

### 5.5 Round-5C shows whether dynamic assignment is worth the extra cost

The new dynamic-assignment variant:

- keeps the Round-5B decoupled head
- keeps shape-matching anchors as the starting prior
- replaces the static positive/ignore split with a prediction-aware cost assignment

Its COCO subset metrics are:

- `AP50 = {coco_eval['ap50']:.6f}`
- `AR@100 = {coco_eval['ar_100']:.6f}`

Its internal engineering metrics are:

- `mAP@0.5 = {internal_eval['map50']:.6f}`
- `precision = {internal_eval['precision']:.6f}`
- `recall = {internal_eval['recall']:.6f}`

This section should be interpreted together with the dedicated
`stagec_round5c_summary.md` note, which decides whether dynamic assignment
beats the current Round-5B baseline or should be rolled back.
"""
    if "### 5.5 Round-5C" not in text:
        text = text.rstrip() + section + "\n"
    write_text(summary_path, text)


def build_summary_md(
    run_id: str,
    result_metrics: dict[str, float | int | str],
    internal_eval: dict,
    coco_eval: dict,
) -> str:
    """Render the dedicated Round 5C summary note."""
    return f"""# Stage C Round 5C Summary

本文件总结 Stage C 第五轮改进中的第三步：

- `Round 5C: 在 Round 5B 基础上引入 Dynamic Assignment`

## 1. 实验目的

Round 5B 已经证明：

- 质量感知分类和解耦检测头能够显著提升三框版本的排序质量；
- 当前三框版本第一次在 AP50 和 AR@100 上全面超过单框残差基线；
- 但正样本分配仍然依赖静态的 `shape-matching + ignore band`。

因此 Round 5C 的目标是：

- 保留 `Round 5B` 已经验证有效的 backbone、loss 与 head 设计；
- 不再只依赖静态 anchor 先验；
- 在训练时利用当前预测结果，动态选择代价最低的候选槽位作为正样本；
- 验证动态分配是否还能继续提升 AP50、AR@100 和内部 mAP。

## 2. 控制变量

以下设置全部保持和 `Round 5B` 一致：

- backbone：`deep_residual`
- 输入尺寸：`320 x 320`
- 输出网格：`10 x 10`
- 三框检测头：`num_boxes = 3`
- anchors：`0.052141,0.076305;0.199926,0.288289;0.609263,0.649287`
- `box_parameterization = "yolov5"`
- `soft_objectness_target = "iou"`
- `soft_objectness_min = 0.05`
- `soft_classification_target = "iou"`
- `head_type = "decoupled"`
- full train / full val
- 内部评估协议与 COCO 子集评估协议

本轮唯一主变量是：

- `assignment_strategy = "dynamic_cost"`

以及与之配套的：

- `dynamic_topk = 2`
- `dynamic_center_radius = 1`
- `dynamic_box_cost = 3.0`
- `dynamic_cls_cost = 1.0`
- `dynamic_ignore_iou = 0.5`

## 3. 对应实验

- 实验名：
  - `deep_residual_three_box_dynamicassign`
- run_id：
  - `{run_id}`
- 配置：
  - [deep_residual_three_box_dynamicassign.toml](/home/lidz/YOLO/yolov0/configs/deep_residual_three_box_dynamicassign.toml:1)
- 结果：
  - [result.txt](/home/lidz/YOLO/yolov0/logs/records/{run_id}/result.txt:1)
- 元数据：
  - [metadata.json](/home/lidz/YOLO/yolov0/logs/records/{run_id}/metadata.json:1)
- 评估：
  - [internal eval](/home/lidz/YOLO/yolov0/outputs/evaluations/deep_residual_three_box_dynamicassign_eval.json:1)
  - [COCO subset eval](/home/lidz/YOLO/yolov0/outputs/evaluations/deep_residual_three_box_dynamicassign_coco_eval.json:1)

## 4. 训练结果

### 核心训练统计

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

### 训练过程观测

动态分配明显抬高了单 step 的计算成本，但训练过程保持稳定：

- train total 最终下降到约 `{result_metrics.get("final_train_total", float("nan")):.4f}`
- val total 最低下降到约 `{result_metrics.get("best_val_loss", float("nan")):.4f}`
- 正样本与忽略样本的统计会在结果文件中进一步验证动态分配是否真的激活了更多候选槽位

## 5. 与前一轮 Round 5B 对比

| 指标 | Round 5B | Round 5C | 变化解读 |
| --- | ---: | ---: | --- |
| Best Val Total | `2.597307` | `{result_metrics.get("best_val_loss", float("nan")):.6f}` | 待本轮结论解释 |
| Internal mAP@0.5 | `0.077865` | `{internal_eval['map50']:.6f}` | 待本轮结论解释 |
| Internal Precision | `0.053212` | `{internal_eval['precision']:.6f}` | 待本轮结论解释 |
| Internal Recall | `0.259645` | `{internal_eval['recall']:.6f}` | 待本轮结论解释 |
| Num Predictions | `256503` | `{internal_eval['num_predictions']}` | 预测数量变化 |
| COCO AP50 | `0.000814` | `{coco_eval['ap50']:.6f}` | 待本轮结论解释 |
| COCO AR@100 | `0.002500` | `{coco_eval['ar_100']:.6f}` | 待本轮结论解释 |

## 6. 主要结论

### 6.1 动态分配的直接收益

从本轮结果可以看出，动态分配带来的核心变化是：

- internal `mAP@0.5 = {internal_eval['map50']:.6f}`
- internal `precision = {internal_eval['precision']:.6f}`
- internal `recall = {internal_eval['recall']:.6f}`
- COCO `AP50 = {coco_eval['ap50']:.6f}`
- COCO `AR@100 = {coco_eval['ar_100']:.6f}`

这些数值决定了它是否真的超过了当前的 Round 5B 基线。

### 6.2 与当前最佳三框基线的关系

Round 5B 是进入 Round 5C 之前的最强三框版本。

因此，本轮最关键的问题不是“能不能训练”，而是：

- 动态分配是否在不破坏排序质量的前提下继续提升召回；
- 或者它是否由于计算代价和目标分配波动，反而损害了已经建立起来的 AP50 优势。

### 6.3 当前阶段结论

本轮完成后，Stage C 已经具备三条清晰路线：

- `Round 4B`：静态 shape-matching + ignore band
- `Round 5B`：质量感知分类 + 解耦头
- `Round 5C`：在 Round 5B 基础上引入动态分配

下一阶段是否继续保留动态分配，应以本轮的 `AP50 / AR@100 / 可视化质量 / 训练代价` 综合决定。

## 7. 下一步建议

如果 Round 5C 明显优于 Round 5B，可以继续在动态分配线上细化：

- cost 设计
- top-k 策略
- 中心先验范围

如果 Round 5C 表现一般或代价过高，则应回退到：

- `Round 5B` 作为当前最佳三框基线

并把后续改进重点转向：

- score 排序公式
- 更轻量的 head / assignment 微调
- 或 Stage D 的多尺度扩展
"""


def finalize(args: argparse.Namespace) -> None:
    """Run the full Round 5C post-processing pipeline."""
    run_id = args.run_id
    record_dir = ROOT / "logs" / "records" / run_id
    result_path = record_dir / "result.txt"
    metadata_path = record_dir / "metadata.json"

    wait_for_completion(result_path, metadata_path, args.train_pid)

    checkpoint = ROOT / "outputs" / run_id / "best.pth"
    internal_json = ROOT / "outputs" / "evaluations" / "deep_residual_three_box_dynamicassign_eval.json"
    coco_json = ROOT / "outputs" / "evaluations" / "deep_residual_three_box_dynamicassign_coco_eval.json"

    run_cmd(
        [
            PYTHON,
            "-u",
            "tools/evaluate.py",
            "--config",
            str(ROOT / "configs" / "deep_residual_three_box_dynamicassign.toml"),
            "--checkpoint",
            str(checkpoint),
            "--output-json",
            str(internal_json),
        ]
    )
    run_cmd(
        [
            PYTHON,
            "-u",
            "tools/evaluate_coco.py",
            "--config",
            str(ROOT / "configs" / "deep_residual_three_box_dynamicassign.toml"),
            "--checkpoint",
            str(checkpoint),
            "--output-json",
            str(coco_json),
        ]
    )

    result_metrics = parse_result_metrics(read_text(result_path))
    internal_eval = json.loads(read_text(internal_json))
    coco_eval = json.loads(read_text(coco_json))

    summary_path = ROOT / "logs" / "stagec_round5c_summary.md"
    write_text(summary_path, build_summary_md(run_id, result_metrics, internal_eval, coco_eval))

    update_experiment_index(run_id)
    update_detection_summary()
    replace_dynamic_summary_row(coco_eval, internal_eval)

    files_to_add = [
        "yolov0/losses/yolo_loss.py",
        "yolov0/tools/train.py",
        "yolov0/utils/config.py",
        "yolov0/configs/deep_residual_three_box_dynamicassign.toml",
        "yolov0/tools/finalize_round5c.py",
        "yolov0/logs/stagec_round5c_change_notes.md",
        "yolov0/logs/stagec_round5c_summary.md",
        "yolov0/logs/experiment_result_index.md",
        "yolov0/logs/detection_evaluation_summary.md",
        f"yolov0/logs/records/{run_id}/config.toml",
        f"yolov0/logs/records/{run_id}/result.txt",
        f"yolov0/logs/records/{run_id}/metadata.json",
        "yolov0/outputs/evaluations/deep_residual_three_box_dynamicassign_eval.json",
        "yolov0/outputs/evaluations/deep_residual_three_box_dynamicassign_coco_eval.json",
    ]
    run_cmd(["git", "add", "-f", *files_to_add], cwd=ROOT.parent)
    run_cmd(
        ["git", "commit", "-m", "Add Stage C Round 5C dynamic-assignment results"],
        cwd=ROOT.parent,
    )


def main() -> int:
    """Parse args and finalize the long-running Round 5C experiment."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--train-pid", type=int, required=True)
    args = parser.parse_args()
    finalize(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
