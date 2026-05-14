"""Runtime configuration for the Flask inference backend."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve_path(raw_path: str | None, default: str) -> Path:
    path = Path(raw_path or default).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


@dataclass(frozen=True)
class BackendSettings:
    """Resolved backend settings derived from environment variables."""

    model_format: str
    config_path: Path
    checkpoint_path: Path
    onnx_model_path: Path
    torchscript_model_path: Path
    device: str
    metadata_path: Path
    annotation_dir: Path
    max_upload_bytes: int
    max_image_pixels: int
    max_top_k: int
    preload_model: bool
    use_fp16: bool
    host: str
    port: int
    debug: bool


def load_settings() -> BackendSettings:
    """Load environment-driven backend settings."""
    return BackendSettings(
        model_format=os.getenv("YOLO_BACKEND_MODEL_FORMAT", "torchscript").strip().lower(),
        config_path=_resolve_path(
            os.getenv("YOLO_BACKEND_CONFIG"),
            "configs/dual_scale_three_box_coco_only_noobj1_416_basic_aug_scale_jitter_50_lr7e4.toml",
        ),
        checkpoint_path=_resolve_path(
            os.getenv("YOLO_BACKEND_CHECKPOINT"),
            "outputs/dual_scale_three_box_coco_only_noobj1_416_basic_aug_scale_jitter_50_lr7e4_ddp_20260512_130823/best.pth",
        ),
        onnx_model_path=_resolve_path(os.getenv("YOLO_BACKEND_ONNX_MODEL"), "exports/model.onnx"),
        torchscript_model_path=_resolve_path(
            os.getenv("YOLO_BACKEND_TORCHSCRIPT_MODEL"),
            "exports/checkpoint8/best_yolofinal_416_lr7e4.torchscript.pt",
        ),
        device=os.getenv("YOLO_BACKEND_DEVICE", "auto").strip(),
        metadata_path=_resolve_path(
            os.getenv("YOLO_BACKEND_METADATA"),
            "../DataSet/Unified/metadata/class_maps.json",
        ),
        annotation_dir=_resolve_path(os.getenv("YOLO_BACKEND_ANNOTATION_DIR"), "backend/annotations"),
        max_upload_bytes=int(float(os.getenv("YOLO_BACKEND_MAX_UPLOAD_MB", "20")) * 1024 * 1024),
        max_image_pixels=int(os.getenv("YOLO_BACKEND_MAX_IMAGE_PIXELS", "25000000")),
        max_top_k=int(os.getenv("YOLO_BACKEND_MAX_TOP_K", "500")),
        preload_model=os.getenv("YOLO_BACKEND_PRELOAD_MODEL", "0").strip().lower() in {"1", "true", "yes", "on"},
        use_fp16=os.getenv("YOLO_BACKEND_USE_FP16", "0").strip().lower() in {"1", "true", "yes", "on"},
        host=os.getenv("YOLO_BACKEND_HOST", "127.0.0.1").strip(),
        port=int(os.getenv("YOLO_BACKEND_PORT", "5000")),
        debug=os.getenv("YOLO_BACKEND_DEBUG", "0").strip().lower() in {"1", "true", "yes", "on"},
    )
