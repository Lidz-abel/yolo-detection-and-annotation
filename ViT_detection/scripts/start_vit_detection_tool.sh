#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export YOLO_BACKEND_CONFIG="${YOLO_BACKEND_CONFIG:-configs/hybrid_vit_p3p4p5_416_ddp.toml}"
export YOLO_BACKEND_MODEL_FORMAT="${YOLO_BACKEND_MODEL_FORMAT:-pytorch}"
export YOLO_BACKEND_CHECKPOINT="${YOLO_BACKEND_CHECKPOINT:-outputs/hybrid_vit_p3p4p5_416_ddp/best.pth}"

PYTHON_BIN="${PYTHON:-python}"
YOLOV1_PYTHON="/home/lidz/miniconda3/envs/yolov1/bin/python"

if ! "${PYTHON_BIN}" -c "import flask, torch" >/dev/null 2>&1 && [ -x "${YOLOV1_PYTHON}" ]; then
  PYTHON_BIN="${YOLOV1_PYTHON}"
fi

"${PYTHON_BIN}" backend/app.py
