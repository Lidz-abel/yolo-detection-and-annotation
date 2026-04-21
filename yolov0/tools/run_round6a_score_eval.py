#!/usr/bin/env python3
"""Run the formal Round 6A score-tuning evaluation on top of Round 5C.

Round 6A keeps the Round 5C checkpoint fixed and only changes the ranking
formula used during evaluation and visualization. The goal is to test whether
stronger objectness weighting can improve AP50 without retraining.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from data.detection_dataset import DetectionDataset
from models.detector import YOLOv0Baseline
from utils.config import load_config, parse_anchor_string, summarize_config
from utils.experiment import init_run, update_metadata, write_result_summary
from utils.modeling import count_parameters, describe_model_output
from utils.visualization import save_visualization_set


PYTHON = "/home/lidz/anaconda3/envs/yolov1/bin/python"


def parse_args() -> argparse.Namespace:
    """Parse the Round 6A config and source checkpoint inputs."""
    parser = argparse.ArgumentParser(description="Run the formal Round 6A score-tuning evaluation.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "deep_residual_three_box_dynamicassign_scoretune.toml"),
        help="Round 6A config path.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "deep_residual_three_box_dynamicassign_20260420_030501" / "best.pth"),
        help="Source checkpoint from the Round 5C baseline.",
    )
    return parser.parse_args()


def build_device(device_name: str) -> torch.device:
    """Resolve the configured device name into one torch device."""
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def load_state_dict(model, state_dict) -> None:
    """Load weights into wrapped or plain modules uniformly."""
    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)


def read_json(path: Path) -> dict:
    """Read one UTF-8 JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def append_unique_line(path: Path, marker: str, line: str) -> None:
    """Insert one markdown line before a marker if it is not already present."""
    text = path.read_text(encoding="utf-8")
    if line in text:
        return
    if marker in text:
        text = text.replace(marker, line + "\n" + marker, 1)
    else:
        text = text.rstrip() + "\n" + line + "\n"
    path.write_text(text, encoding="utf-8")


def append_unique_row(path: Path, anchor_row: str, new_row: str) -> None:
    """Append one markdown table row after a known anchor row if it is absent."""
    text = path.read_text(encoding="utf-8")
    if new_row in text:
        return
    if anchor_row in text:
        text = text.replace(anchor_row, anchor_row + "\n" + new_row, 1)
    else:
        text = text.rstrip() + "\n" + new_row + "\n"
    path.write_text(text, encoding="utf-8")


def run_cmd(cmd: list[str]) -> None:
    """Run one subprocess from the project root."""
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)


def build_summary(run_id: str, config: dict, internal_eval: dict, coco_eval: dict, checkpoint_path: Path) -> str:
    """Render the dedicated Round 6A markdown summary."""
    eval_cfg = config["evaluation"]
    vis_cfg = config["visualization"]
    return f"""# Stage C Round 6A Summary

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
  - `{checkpoint_path}`

本轮唯一主变量是评分公式：

- `score = obj^alpha * cls^beta`
- 其中：
  - `alpha = {eval_cfg.get('score_alpha', 1.0)}`
  - `beta = {eval_cfg.get('score_beta', 1.0)}`

## 3. 对应实验

- 实验名：
  - `deep_residual_three_box_dynamicassign_scoretune`
- run_id：
  - `{run_id}`
- 配置：
  - [deep_residual_three_box_dynamicassign_scoretune.toml](/home/lidz/YOLO/yolov0/configs/deep_residual_three_box_dynamicassign_scoretune.toml:1)
- 结果：
  - [result.txt](/home/lidz/YOLO/yolov0/logs/records/{run_id}/result.txt:1)
- 元数据：
  - [metadata.json](/home/lidz/YOLO/yolov0/logs/records/{run_id}/metadata.json:1)
- 评估：
  - [internal eval](/home/lidz/YOLO/yolov0/outputs/evaluations/deep_residual_three_box_dynamicassign_scoretune_eval.json:1)
  - [COCO subset eval](/home/lidz/YOLO/yolov0/outputs/evaluations/deep_residual_three_box_dynamicassign_scoretune_coco_eval.json:1)

## 4. 结果

- internal `mAP@0.5 = {internal_eval['map50']:.6f}`
- internal `precision = {internal_eval['precision']:.6f}`
- internal `recall = {internal_eval['recall']:.6f}`
- internal `num_predictions = {int(internal_eval['num_predictions'])}`
- COCO `AP50 = {coco_eval['coco_ap50']:.6f}`
- COCO `AR@100 = {coco_eval['coco_ar100']:.6f}`

可视化使用：

- `score_threshold = {vis_cfg.get('score_threshold', 0.25)}`
- `score_alpha = {vis_cfg.get('score_alpha', 1.0)}`
- `score_beta = {vis_cfg.get('score_beta', 1.0)}`

## 5. 结论

Round 6A 的意义不在于扩大召回，而在于验证：

- 当模型已经学会“找到更多候选框”后
- 更强调 objectness 的排序公式
- 是否能让高质量框自然排到更前面

这轮结果应当与 Round 5C 做直接对比，再决定下一步继续改训练端还是先改动态分配超参数。
"""


def main() -> int:
    """Run the complete Round 6A evaluation, visualization, and bookkeeping flow."""
    args = parse_args()
    config_path = Path(args.config).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    config = load_config(config_path)

    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["train"]
    eval_cfg = config["evaluation"]
    vis_cfg = config["visualization"]
    anchors = parse_anchor_string(model_cfg.get("anchors"))
    num_boxes = int(model_cfg.get("num_boxes", 1))
    device = build_device(str(train_cfg["device"]))

    run_info = init_run(PROJECT_ROOT, config_path, config)
    update_metadata(
        run_info["metadata_path"],
        status="running",
        stage="stagec_round6a_score_eval",
        source_checkpoint=str(checkpoint_path),
    )

    eval_json = PROJECT_ROOT / "outputs" / "evaluations" / "deep_residual_three_box_dynamicassign_scoretune_eval.json"
    coco_json = PROJECT_ROOT / "outputs" / "evaluations" / "deep_residual_three_box_dynamicassign_scoretune_coco_eval.json"

    run_cmd(
        [
            PYTHON,
            "-u",
            "tools/evaluate.py",
            "--config",
            str(config_path),
            "--checkpoint",
            str(checkpoint_path),
            "--score-threshold",
            str(float(eval_cfg.get("score_threshold", 0.05))),
            "--score-alpha",
            str(float(eval_cfg.get("score_alpha", 1.0))),
            "--score-beta",
            str(float(eval_cfg.get("score_beta", 1.0))),
            "--top-k",
            str(int(eval_cfg.get("top_k", 100))),
            "--nms-iou-threshold",
            str(float(eval_cfg.get("nms_iou_threshold", 0.5))),
            "--map-iou-threshold",
            str(float(eval_cfg.get("map_iou_threshold", 0.5))),
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
            str(checkpoint_path),
            "--score-threshold",
            str(float(eval_cfg.get("score_threshold", 0.05))),
            "--score-alpha",
            str(float(eval_cfg.get("score_alpha", 1.0))),
            "--score-beta",
            str(float(eval_cfg.get("score_beta", 1.0))),
            "--top-k",
            str(int(eval_cfg.get("top_k", 100))),
            "--nms-iou-threshold",
            str(float(eval_cfg.get("nms_iou_threshold", 0.5))),
            "--output-json",
            str(coco_json),
        ]
    )

    val_dataset = DetectionDataset(
        manifest_path=data_cfg["val_manifest"],
        image_size=int(data_cfg["image_size"]),
        grid_size=int(data_cfg["grid_size"]),
        num_classes=int(data_cfg["num_classes"]),
        num_boxes=num_boxes,
        anchors=anchors,
        anchor_positive_iou=float(model_cfg.get("anchor_positive_iou", 0.25)),
        anchor_ignore_iou=float(model_cfg.get("anchor_ignore_iou", 0.5)),
        anchor_match_metric=str(model_cfg.get("anchor_match_metric", "iou")),
        anchor_shape_ratio=float(model_cfg.get("anchor_shape_ratio", 4.0)),
        anchor_ignore_shape_ratio=model_cfg.get("anchor_ignore_shape_ratio"),
        max_samples=int(data_cfg["val_max_samples"]),
    )
    model = YOLOv0Baseline(
        num_classes=int(data_cfg["num_classes"]),
        model_name=str(model_cfg["name"]),
        width_mult=float(model_cfg["width_mult"]),
        depth_mult=float(model_cfg["depth_mult"]),
        use_residual=bool(model_cfg["use_residual"]),
        num_boxes=num_boxes,
        head_type=str(model_cfg.get("head_type", "shared")),
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    load_state_dict(model, checkpoint)

    visualization_dir = run_info["output_dir"] / "visualizations"
    save_visualization_set(
        model=model,
        dataset=val_dataset,
        output_dir=visualization_dir,
        device=device,
        num_classes=int(data_cfg["num_classes"]),
        num_boxes=num_boxes,
        anchors=anchors,
        box_parameterization=str(model_cfg.get("box_parameterization", "legacy")),
        max_samples=int(vis_cfg.get("max_samples", 4)),
        score_threshold=float(vis_cfg.get("score_threshold", 0.25)),
        top_k=int(vis_cfg.get("top_k", 12)),
        score_alpha=float(vis_cfg.get("score_alpha", 1.0)),
        score_beta=float(vis_cfg.get("score_beta", 1.0)),
    )

    internal_eval = read_json(eval_json)
    coco_eval = read_json(coco_json)
    param_stats = count_parameters(model)
    output_shape = describe_model_output(model, int(data_cfg["image_size"]), device)
    summary_path = PROJECT_ROOT / "logs" / "stagec_round6a_summary.md"
    summary_path.write_text(
        build_summary(run_info["run_id"], config, internal_eval, coco_eval, checkpoint_path),
        encoding="utf-8",
    )

    result_lines = [
        "stage = stagec_round6a_score_eval",
        "status = completed",
        f"run_id = {run_info['run_id']}",
        f"source_checkpoint = {checkpoint_path}",
        f"model_output_shape = {output_shape}",
        f"parameter_total = {param_stats['total']}",
        f"parameter_trainable = {param_stats['trainable']}",
        f"output_dir = {run_info['output_dir']}",
        f"visualization_dir = {visualization_dir}",
        "",
        *summarize_config(config),
        "",
        "[internal evaluation]",
        f"precision = {internal_eval['precision']:.6f}",
        f"recall = {internal_eval['recall']:.6f}",
        f"map50 = {internal_eval['map50']:.6f}",
        f"num_predictions = {int(internal_eval['num_predictions'])}",
        "",
        "[coco subset evaluation]",
        f"coco_ap = {coco_eval['coco_ap']:.6f}",
        f"coco_ap50 = {coco_eval['coco_ap50']:.6f}",
        f"coco_ap75 = {coco_eval['coco_ap75']:.6f}",
        f"coco_ar100 = {coco_eval['coco_ar100']:.6f}",
    ]
    write_result_summary(run_info["result_path"], result_lines)
    update_metadata(
        run_info["metadata_path"],
        status="completed",
        eval_json=str(eval_json),
        coco_eval_json=str(coco_json),
        visualization_dir=str(visualization_dir),
    )

    index_path = PROJECT_ROOT / "logs" / "experiment_result_index.md"
    index_row = (
        f"| deep_residual_three_box_dynamicassign_scoretune | `{run_info['run_id']}` | "
        f"[deep_residual_three_box_dynamicassign_scoretune.toml](/home/lidz/YOLO/yolov0/configs/deep_residual_three_box_dynamicassign_scoretune.toml:1) | "
        f"[result.txt](/home/lidz/YOLO/yolov0/logs/records/{run_info['run_id']}/result.txt:1) | "
        f"[metadata.json](/home/lidz/YOLO/yolov0/logs/records/{run_info['run_id']}/metadata.json:1) | "
        f"[eval.json](/home/lidz/YOLO/yolov0/outputs/evaluations/deep_residual_three_box_dynamicassign_scoretune_eval.json:1), "
        f"[coco_eval.json](/home/lidz/YOLO/yolov0/outputs/evaluations/deep_residual_three_box_dynamicassign_scoretune_coco_eval.json:1) | "
        f"[outputs](/home/lidz/YOLO/yolov0/outputs/{run_info['run_id']}:1) |"
    )
    append_unique_line(index_path, "\n## 使用规则\n", index_row)

    detection_summary_path = PROJECT_ROOT / "logs" / "detection_evaluation_summary.md"
    json_bullet = (
        "- `/home/lidz/YOLO/yolov0/outputs/evaluations/"
        "deep_residual_three_box_dynamicassign_scoretune_coco_eval.json`"
    )
    append_unique_line(detection_summary_path, "\n### 4.1 Result Table\n", json_bullet)
    anchor_row = (
        "| `deep_residual` + three-box + dynamic-assign | 3 | 28.92M | 42.66G | 98.41 | 0.000299 | 0.001150 | 0.000120 | 0.003097 |"
    )
    table_row = (
        "| `deep_residual` + three-box + dynamic-assign + score-tune | 3 | "
        f"{internal_eval['params_total'] / 1_000_000:.2f}M | {internal_eval['total_gflops']:.2f}G | {internal_eval['images_per_second']:.2f} | "
        f"{coco_eval['coco_ap']:.6f} | {coco_eval['coco_ap50']:.6f} | {coco_eval['coco_ap75']:.6f} | {coco_eval['coco_ar100']:.6f} |"
    )
    append_unique_row(detection_summary_path, anchor_row, table_row)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
