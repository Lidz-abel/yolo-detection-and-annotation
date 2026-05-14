#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON:-python}"
YOLOV1_PYTHON="/home/lidz/miniconda3/envs/yolov1/bin/python"

if ! "${PYTHON_BIN}" -c "import torch" >/dev/null 2>&1 && [ -x "${YOLOV1_PYTHON}" ]; then
  PYTHON_BIN="${YOLOV1_PYTHON}"
fi

"${PYTHON_BIN}" tools/smoke_dual_scale_three_box.py \
  --config configs/hybrid_vit_p3p4p5_416_ddp.toml

"${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE:-8}" \
  tools/train_ddp.py \
  --config configs/hybrid_vit_p3p4p5_416_ddp.toml
