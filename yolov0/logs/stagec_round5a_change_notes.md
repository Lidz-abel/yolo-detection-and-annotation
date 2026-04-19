# Stage C Round 5A Change Notes

## Goal

Round 5A targets the current Stage C scoring bottleneck. The aim is to keep
Round 4B's recall-oriented matching behavior while reducing high-score low-IoU
boxes.

## Controlled Variables

The following parts stay fixed relative to Round 4B:

- backbone: `deep_residual`
- input size: `320 x 320`
- output grid: `10 x 10`
- detection head: `3 x (80 + 5)`
- box parameterization: `yolov5`
- matching: `shape_ratio`
- ignore band: `anchor_ignore_shape_ratio = 4.0`

## Main Change

The classification target becomes quality-aware:

- before: positive class target = `1.0`
- now: positive class target = current matched IoU

This means low-quality matched boxes receive weaker class reward, which should
reduce the final score of duplicate or low-quality boxes.

## Code Touch Points

- `losses/yolo_loss.py`
- `engine/trainer.py`
- `tools/train.py`
- `utils/config.py`
- `configs/deep_residual_three_box_qualitycls.toml`
