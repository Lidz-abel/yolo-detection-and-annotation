"""Export a trained detector checkpoint to TorchScript and/or ONNX."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from models.detector import YOLOv0Baseline
from utils.config import load_config, parse_string_list


DEFAULT_CONFIG = "configs/dual_scale_three_box_coco_only_noobj1_416_basic_aug_scale_jitter_50_lr7e4.toml"
DEFAULT_CHECKPOINT = (
    "outputs/dual_scale_three_box_coco_only_noobj1_416_basic_aug_scale_jitter_50_lr7e4_ddp_20260512_130823/best.pth"
)


class TupleOutputWrapper(nn.Module):
    """Return multi-scale detector outputs as an ordered tuple for export."""

    def __init__(self, model: nn.Module, output_names: list[str]):
        super().__init__()
        self.model = model
        self.output_names = output_names

    def forward(self, images: torch.Tensor):
        outputs = self.model(images)
        if isinstance(outputs, dict):
            return tuple(outputs[name] for name in self.output_names)
        return (outputs,)


def parse_args():
    parser = argparse.ArgumentParser(description="Export YOLO detector checkpoint.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", default="exports/checkpoint8")
    parser.add_argument("--prefix", default="best_yolofinal_416")
    parser.add_argument("--formats", default="torchscript,onnx", help="Comma list: torchscript,onnx")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--strict-onnx", action="store_true", help="Fail if ONNX export deps are missing.")
    return parser.parse_args()


def build_model(config: dict, checkpoint_path: Path, device: torch.device) -> nn.Module:
    data_cfg = config["data"]
    model_cfg = config["model"]
    feature_levels = parse_string_list(model_cfg.get("feature_levels"))
    model = YOLOv0Baseline(
        num_classes=int(data_cfg["num_classes"]),
        model_name=str(model_cfg["name"]),
        width_mult=float(model_cfg["width_mult"]),
        depth_mult=float(model_cfg["depth_mult"]),
        use_residual=bool(model_cfg["use_residual"]),
        num_boxes=int(model_cfg.get("num_boxes", 1)),
        head_type=str(model_cfg.get("head_type", "shared")),
        neck_type=str(model_cfg.get("neck_type", "none")),
        feature_levels=feature_levels,
    ).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model


def metadata_payload(args, config: dict, output_names: list[str], artifacts: dict) -> dict:
    image_size = int(config["data"]["image_size"])
    input_names = ["images"]
    dynamic_axes = {"images": {0: "batch"}}
    for name in output_names:
        dynamic_axes[name] = {0: "batch"}
    return {
        "config_path": str(Path(args.config).resolve()),
        "checkpoint_path": str(Path(args.checkpoint).resolve()),
        "image_size": image_size,
        "input_names": input_names,
        "output_names": output_names,
        "dynamic_axes": dynamic_axes,
        "opset": int(args.opset),
        "formats_requested": [item.strip() for item in args.formats.split(",") if item.strip()],
        "artifacts": artifacts,
        "notes": [
            "Outputs are raw prediction tensors in feature-level order.",
            "Post-processing stays in Python to keep PyTorch, TorchScript, and ONNX comparisons controlled.",
            "Dynamic axes are batch-only; image H/W are fixed by the 416-grid training config.",
        ],
    }


def export_torchscript(wrapper: nn.Module, dummy: torch.Tensor, output_path: Path) -> dict:
    with torch.inference_mode():
        traced = torch.jit.trace(wrapper, dummy, strict=False)
        traced = torch.jit.freeze(traced.eval())
        traced.save(str(output_path))
    return {"path": str(output_path), "bytes": output_path.stat().st_size}


def export_onnx(
    wrapper: nn.Module,
    dummy: torch.Tensor,
    output_path: Path,
    output_names: list[str],
    opset: int,
    strict: bool,
) -> dict:
    try:
        import onnx  # noqa: F401
    except ImportError as exc:
        message = "onnx package is not installed; skipping ONNX export."
        if strict:
            raise RuntimeError(message) from exc
        return {"path": str(output_path), "skipped": True, "reason": message}

    dynamic_axes = {"images": {0: "batch"}}
    for name in output_names:
        dynamic_axes[name] = {0: "batch"}
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            dummy,
            str(output_path),
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=["images"],
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )
    return {"path": str(output_path), "bytes": output_path.stat().st_size}


def max_abs_diff(reference, candidate) -> dict[str, float]:
    result = {}
    for index, (ref, got) in enumerate(zip(reference, candidate)):
        diff = (ref.detach().cpu() - got.detach().cpu()).abs()
        result[f"output_{index}_max_abs_diff"] = float(diff.max().item())
        result[f"output_{index}_mean_abs_diff"] = float(diff.mean().item())
    return result


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(Path(args.config).resolve())
    checkpoint_path = Path(args.checkpoint).resolve()
    device = torch.device(args.device)
    image_size = int(config["data"]["image_size"])
    feature_levels = parse_string_list(config["model"].get("feature_levels")) or ["output"]
    output_names = feature_levels

    model = build_model(config, checkpoint_path, device)
    wrapper = TupleOutputWrapper(model, output_names).to(device).eval()
    dummy = torch.randn(int(args.batch_size), 3, image_size, image_size, device=device)
    with torch.inference_mode():
        reference_outputs = wrapper(dummy)

    artifacts: dict[str, dict] = {}
    requested = {item.strip().lower() for item in args.formats.split(",") if item.strip()}
    if "torchscript" in requested:
        ts_path = output_dir / f"{args.prefix}.torchscript.pt"
        artifacts["torchscript"] = export_torchscript(wrapper, dummy, ts_path)
        loaded = torch.jit.load(str(ts_path), map_location=device).eval()
        with torch.inference_mode():
            artifacts["torchscript"]["correctness"] = max_abs_diff(reference_outputs, loaded(dummy))

    if "onnx" in requested:
        onnx_path = output_dir / f"{args.prefix}.onnx"
        artifacts["onnx"] = export_onnx(
            wrapper=wrapper,
            dummy=dummy,
            output_path=onnx_path,
            output_names=output_names,
            opset=int(args.opset),
            strict=bool(args.strict_onnx),
        )

    payload = metadata_payload(args, config, output_names, artifacts)
    metadata_path = output_dir / f"{args.prefix}.export_metadata.json"
    metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
