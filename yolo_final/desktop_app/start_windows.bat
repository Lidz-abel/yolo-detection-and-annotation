@echo off
setlocal

cd /d "%~dp0\.."

set YOLO_BACKEND_MODEL_FORMAT=torchscript
set YOLO_BACKEND_DEVICE=auto
set YOLO_BACKEND_CONFIG=configs\dual_scale_three_box_coco_only_noobj1_416_basic_aug_scale_jitter_50_lr7e4.toml
set YOLO_BACKEND_CHECKPOINT=outputs\dual_scale_three_box_coco_only_noobj1_416_basic_aug_scale_jitter_50_lr7e4_ddp_20260512_130823\best.pth
set YOLO_BACKEND_TORCHSCRIPT_MODEL=exports\checkpoint8\best_yolofinal_416_lr7e4.torchscript.pt
set YOLO_BACKEND_METADATA=..\DataSet\Unified\metadata\class_maps.json

python desktop_app\yolo_desktop_app_v2.py

if errorlevel 1 (
  echo.
  echo YOLO desktop app failed to start.
  echo Please check that Python, torch, Pillow and numpy are installed.
  pause
)
