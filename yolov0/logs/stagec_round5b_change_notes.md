# Stage C Round 5B Change Notes

## Goal

Round 5B targets the next suspected Stage C bottleneck:

- classification and box/objectness are still produced by one shared head
- quality-aware cls supervision has already shown that ranking matters
- but Round 5A also showed that stronger cls filtering alone can suppress recall too much

## Controlled Variables

The following parts stay fixed relative to Round 5A:

- backbone: `deep_residual`
- input size: `320 x 320`
- output grid: `10 x 10`
- detection head outputs: `3 x (80 + 5)`
- box parameterization: `yolov5`
- matching: `shape_ratio`
- ignore band: `anchor_ignore_shape_ratio = 4.0`
- quality-aware cls target: `soft_classification_target = "iou"`

## Main Change

The shared detection head is replaced with a decoupled head:

- one branch predicts `cls`
- one branch predicts `box + obj`

This keeps the detector interface unchanged while separating the final feature
transformations used by classification and regression/objectness.

## Code Touch Points

- `models/head.py`
- `models/detector.py`
- `tools/train.py`
- `utils/config.py`
- `configs/deep_residual_three_box_qualitycls_decoupled.toml`
