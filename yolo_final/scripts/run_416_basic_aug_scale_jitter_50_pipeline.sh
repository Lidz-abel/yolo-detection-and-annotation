#!/usr/bin/env bash
set -o pipefail

cd /home/lidz/YOLO/yolo_final || exit 1

source /home/lidz/miniconda3/etc/profile.d/conda.sh
conda activate yolov1

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

CONFIG="configs/dual_scale_three_box_coco_only_noobj1_416_basic_aug_scale_jitter_50.toml"
RUN_PREFIX="dual_scale_three_box_coco_only_noobj1_416_basic_aug_scale_jitter_50_ddp"
LOG_DIR="logs/training_tmux"
mkdir -p "${LOG_DIR}" logs/non_training_sweeps outputs/evaluations
LOG="${LOG_DIR}/${RUN_PREFIX}_$(date +%Y%m%d_%H%M%S).log"

echo "session=train_416_basic_aug_scale_jitter_50" | tee -a "${LOG}"
echo "config=${CONFIG}" | tee -a "${LOG}"
echo "log=${LOG}" | tee -a "${LOG}"
echo "stage=train_ddp_8gpu" | tee -a "${LOG}"

torchrun --nproc_per_node=8 tools/train_ddp.py --config "${CONFIG}" 2>&1 | tee -a "${LOG}"
TRAIN_STATUS=${PIPESTATUS[0]}
echo "training_exit=${TRAIN_STATUS}" | tee -a "${LOG}"

if [ "${TRAIN_STATUS}" -ne 0 ]; then
  echo "training_failed_skip_eval_sweep_vis_loss" | tee -a "${LOG}"
  exit "${TRAIN_STATUS}"
fi

RUN_DIR=$(ls -dt outputs/${RUN_PREFIX}_* | head -n 1)
RUN_ID=$(basename "${RUN_DIR}")
BEST_CKPT="${RUN_DIR}/best.pth"
RECORD_DIR="logs/records/${RUN_ID}"
RESULT_TXT="${RECORD_DIR}/result.txt"
echo "postrun_run_id=${RUN_ID}" | tee -a "${LOG}"
echo "best_checkpoint=${BEST_CKPT}" | tee -a "${LOG}"

echo "stage=coco_eval_full_val2017" | tee -a "${LOG}"
CUDA_VISIBLE_DEVICES=0 python tools/evaluate_coco.py \
  --config "${CONFIG}" \
  --checkpoint "${BEST_CKPT}" \
  --score-threshold 0.05 \
  --score-alpha 2.0 \
  --score-beta 1.0 \
  --top-k 100 \
  --nms-iou-threshold 0.5 \
  --output-json "outputs/evaluations/${RUN_ID}_coco_eval.json" \
  2>&1 | tee -a "${LOG}"
EVAL_STATUS=${PIPESTATUS[0]}
echo "coco_eval_exit=${EVAL_STATUS}" | tee -a "${LOG}"

if [ "${EVAL_STATUS}" -ne 0 ]; then
  echo "coco_eval_failed_skip_sweep_vis_loss" | tee -a "${LOG}"
  exit "${EVAL_STATUS}"
fi

echo "stage=sweep_500" | tee -a "${LOG}"
CUDA_VISIBLE_DEVICES=0 python tools/sweep_coco_checkpoint.py \
  --config "${CONFIG}" \
  --checkpoint "${BEST_CKPT}" \
  --max-samples 500 \
  --score-alphas 1.5,2.0 \
  --score-betas 1.0 \
  --score-thresholds 0.03,0.05,0.1,0.2 \
  --top-ks 20,50,100,200 \
  --nms-iou-thresholds 0.4,0.5,0.6 \
  --output-json "outputs/evaluations/${RUN_ID}_sweep_500.json" \
  2>&1 | tee -a "${LOG}"
SWEEP_STATUS=${PIPESTATUS[0]}
echo "sweep_exit=${SWEEP_STATUS}" | tee -a "${LOG}"

if [ "${SWEEP_STATUS}" -ne 0 ]; then
  echo "sweep_failed_skip_vis_loss" | tee -a "${LOG}"
  exit "${SWEEP_STATUS}"
fi

echo "stage=pred_only_vis12" | tee -a "${LOG}"
CUDA_VISIBLE_DEVICES=0 python tools/diagnose_coco_checkpoint.py \
  --config "${CONFIG}" \
  --checkpoint "${BEST_CKPT}" \
  --max-samples 12 \
  --vis-samples 12 \
  --vis-score-threshold 0.50 \
  --vis-top-k 10 \
  --vis-nms-iou-threshold 0.5 \
  --vis-score-alpha 2.0 \
  --vis-score-beta 1.0 \
  --vis-hide-gt \
  --vis-dir "outputs/evaluations/${RUN_ID}_display_pred_only_vis12" \
  --output-json "outputs/evaluations/${RUN_ID}_display_pred_only_vis12.json" \
  2>&1 | tee -a "${LOG}"
VIS_STATUS=${PIPESTATUS[0]}
echo "pred_only_vis12_exit=${VIS_STATUS}" | tee -a "${LOG}"

if [ "${VIS_STATUS}" -ne 0 ]; then
  echo "vis_failed_skip_loss_plot" | tee -a "${LOG}"
  exit "${VIS_STATUS}"
fi

echo "stage=loss_curves" | tee -a "${LOG}"
python tools/plot_loss_curves.py \
  --result-txt "${RESULT_TXT}" \
  --output-png "outputs/evaluations/${RUN_ID}_loss_curves.png" \
  --title "${RUN_ID} Loss Curves" \
  2>&1 | tee -a "${LOG}"
LOSS_STATUS=${PIPESTATUS[0]}
echo "loss_curves_exit=${LOSS_STATUS}" | tee -a "${LOG}"

exit "${LOSS_STATUS}"
