#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-int8-cpu}"

export YOLO_BACKEND_CONFIG="${YOLO_BACKEND_CONFIG:-configs/dual_scale_three_box_coco_only_noobj1_416_basic_aug_scale_jitter_50_lr7e4.toml}"
export YOLO_BACKEND_HOST="${YOLO_BACKEND_HOST:-127.0.0.1}"
export YOLO_BACKEND_PORT="${YOLO_BACKEND_PORT:-5000}"
export YOLO_BACKEND_MAX_TOP_K="${YOLO_BACKEND_MAX_TOP_K:-500}"
export YOLO_BACKEND_PRELOAD_MODEL="${YOLO_BACKEND_PRELOAD_MODEL:-1}"

case "${MODE}" in
  fp32-gpu|fp32|torchscript)
    export YOLO_BACKEND_MODEL_FORMAT=torchscript
    export YOLO_BACKEND_DEVICE="${YOLO_BACKEND_DEVICE:-auto}"
    export YOLO_BACKEND_USE_FP16="${YOLO_BACKEND_USE_FP16:-0}"
    export YOLO_BACKEND_TORCHSCRIPT_MODEL="${YOLO_BACKEND_TORCHSCRIPT_MODEL:-exports/checkpoint8/best_yolofinal_416_lr7e4.torchscript.pt}"
    ;;
  int8-cpu|int8)
    export YOLO_BACKEND_MODEL_FORMAT=torchscript
    export YOLO_BACKEND_DEVICE=cpu
    export YOLO_BACKEND_USE_FP16=0
    export YOLO_BACKEND_TORCHSCRIPT_MODEL="${YOLO_BACKEND_TORCHSCRIPT_MODEL:-exports/checkpoint8/best_yolofinal_416_lr7e4_int8_backbone_calib128.torchscript.pt}"
    ;;
  pth|pytorch)
    export YOLO_BACKEND_MODEL_FORMAT=pytorch
    export YOLO_BACKEND_DEVICE="${YOLO_BACKEND_DEVICE:-auto}"
    export YOLO_BACKEND_USE_FP16="${YOLO_BACKEND_USE_FP16:-1}"
    export YOLO_BACKEND_CHECKPOINT="${YOLO_BACKEND_CHECKPOINT:-outputs/dual_scale_three_box_coco_only_noobj1_416_basic_aug_scale_jitter_50_lr7e4_ddp_20260512_130823/best.pth}"
    ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    echo "Usage: scripts/start_yolo_detection_tool.sh [fp32-gpu|int8-cpu|pytorch]" >&2
    exit 2
    ;;
esac

echo "YOLO detection tool"
echo "  mode: ${MODE}"
echo "  format: ${YOLO_BACKEND_MODEL_FORMAT}"
echo "  config: ${YOLO_BACKEND_CONFIG}"
echo "  device: ${YOLO_BACKEND_DEVICE}"
echo "  torchscript: ${YOLO_BACKEND_TORCHSCRIPT_MODEL:-}"
echo "  checkpoint: ${YOLO_BACKEND_CHECKPOINT:-}"
echo "  url: http://${YOLO_BACKEND_HOST}:${YOLO_BACKEND_PORT}/"

python backend/app.py
