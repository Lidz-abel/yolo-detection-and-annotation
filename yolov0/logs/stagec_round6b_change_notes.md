# Stage C Round 6B Change Notes

Round 6B is prepared but not started yet.

Planned goal:

- keep the best dynamic-assignment baseline after the Round 6C comparison;
- improve ranking quality at training time rather than only at inference time;
- reduce extra low-quality boxes without giving up the recall gains already won by Round 5C/6C.

Planned mechanism:

- keep `soft_classification_target = "iou"`;
- replace positive-only BCE classification with a denser varifocal-style loss;
- let positive class targets stay IoU-aware while negative class logits receive a score-dependent penalty.

Initial prepared config:

- [deep_residual_three_box_dynamicassign_topk1_varifocal.toml](/home/lidz/YOLO/yolov0/configs/deep_residual_three_box_dynamicassign_topk1_varifocal.toml:1)

After the Round 6C comparison, the baseline decision changed:

- Round 6C (`dynamic_topk = 1`) is weaker than both Round 5C and Round 6A.
- Therefore Round 6B should **not** be built on top of `topk1`.
- The active Round 6B training config is now:
  - [deep_residual_three_box_dynamicassign_varifocal.toml](/home/lidz/YOLO/yolov0/configs/deep_residual_three_box_dynamicassign_varifocal.toml:1)

This keeps the stronger Round 5C dynamic assignment:

- `dynamic_topk = 2`

and adds:

- `cls_loss_mode = "varifocal"`
- `varifocal_alpha = 0.75`
- `varifocal_gamma = 2.0`

The intent is to improve ranking quality in training without inheriting the
recall/precision regression observed in Round 6C.
