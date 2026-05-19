"""Create a portable folder for the offline Windows YOLO desktop app."""

from __future__ import annotations

import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "desktop_app_dist" / "yolo_windows_desktop"

BEST_CONFIG = PROJECT_ROOT / "configs" / "dual_scale_three_box_coco_only_noobj1_416_basic_aug_scale_jitter_50_lr7e4.toml"
BEST_TORCHSCRIPT = PROJECT_ROOT / "exports" / "checkpoint8" / "best_yolofinal_416_lr7e4.torchscript.pt"
BEST_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "dual_scale_three_box_coco_only_noobj1_416_basic_aug_scale_jitter_50_lr7e4_ddp_20260512_130823"
    / "best.pth"
)
CLASS_MAP = PROJECT_ROOT.parent / "DataSet" / "Unified" / "metadata" / "class_maps.json"


def copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "tests", "annotations")
    shutil.copytree(src, dst, ignore=ignore)


def write_runner(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "@echo off",
                "setlocal",
                'cd /d "%~dp0"',
                "",
                "set YOLO_BACKEND_MODEL_FORMAT=torchscript",
                "set YOLO_BACKEND_DEVICE=auto",
                "set YOLO_BACKEND_CONFIG=configs\\dual_scale_three_box_coco_only_noobj1_416_basic_aug_scale_jitter_50_lr7e4.toml",
                "set YOLO_BACKEND_CHECKPOINT=outputs\\dual_scale_three_box_coco_only_noobj1_416_basic_aug_scale_jitter_50_lr7e4_ddp_20260512_130823\\best.pth",
                "set YOLO_BACKEND_TORCHSCRIPT_MODEL=exports\\checkpoint8\\best_yolofinal_416_lr7e4.torchscript.pt",
                "set YOLO_BACKEND_METADATA=metadata\\class_maps.json",
                "",
                "python desktop_app\\yolo_desktop_app_v2.py",
                "",
                "if errorlevel 1 (",
                "  echo.",
                "  echo YOLO desktop app failed to start.",
                "  echo Please check Python and dependencies: torch, Pillow, numpy.",
                "  pause",
                ")",
                "",
            ]
        ),
        encoding="utf-8",
    )


def normalize_batch_line_endings(root: Path) -> None:
    """Make Windows command scripts friendlier after generating them on Linux."""
    for path in root.rglob("*.bat"):
        text = path.read_text(encoding="utf-8")
        path.write_text(text.replace("\r\n", "\n").replace("\n", "\r\n"), encoding="utf-8")
    for path in root.rglob("*.ps1"):
        text = path.read_text(encoding="utf-8")
        path.write_text(text.replace("\r\n", "\n").replace("\n", "\r\n"), encoding="utf-8")


def main() -> None:
    if OUTPUT_ROOT.exists():
        raise FileExistsError(f"Output already exists, remove it first if needed: {OUTPUT_ROOT}")

    copy_tree(PROJECT_ROOT / "backend", OUTPUT_ROOT / "backend")
    copy_tree(PROJECT_ROOT / "utils", OUTPUT_ROOT / "utils")
    copy_tree(PROJECT_ROOT / "models", OUTPUT_ROOT / "models")
    copy_tree(PROJECT_ROOT / "losses", OUTPUT_ROOT / "losses")
    copy_tree(PROJECT_ROOT / "desktop_app", OUTPUT_ROOT / "desktop_app")
    copy_file(BEST_CONFIG, OUTPUT_ROOT / "configs" / BEST_CONFIG.name)
    copy_file(BEST_TORCHSCRIPT, OUTPUT_ROOT / "exports" / "checkpoint8" / BEST_TORCHSCRIPT.name)
    copy_file(BEST_CHECKPOINT, OUTPUT_ROOT / "outputs" / BEST_CHECKPOINT.parent.name / BEST_CHECKPOINT.name)
    copy_file(CLASS_MAP, OUTPUT_ROOT / "metadata" / "class_maps.json")
    copy_file(PROJECT_ROOT / "desktop_app" / "requirements_windows.txt", OUTPUT_ROOT / "requirements_windows.txt")
    write_runner(OUTPUT_ROOT / "run_yolo_desktop.bat")
    normalize_batch_line_endings(OUTPUT_ROOT)

    print(f"Created portable Windows desktop bundle: {OUTPUT_ROOT}")
    print("Entry point on Windows: run_yolo_desktop.bat")


if __name__ == "__main__":
    main()
