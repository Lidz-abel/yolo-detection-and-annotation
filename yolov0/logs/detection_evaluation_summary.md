# Detection Evaluation Supplement

This note records the formal evaluation supplement added after stages A,
B, and C.

The main purpose of this supplement is:

- to move beyond loss-only comparison
- to add more rigorous efficiency and detection metrics
- to clarify where official-style evaluation is possible and where it is
  not yet semantically clean

## 1. Why Official Evaluation Is Only Applied to the COCO Subset

The unified validation manifest:

- `/home/lidz/YOLO/DataSet/Unified/manifests/all_val.jsonl`

contains both:

- `voc2012`
- `coco2017`

However, these two sources do **not** share the same semantic label space.

Examples:

- VOC label `19` means `tvmonitor`
- COCO label `19` means the contiguous COCO class at index `19`

Therefore, a single official-style metric over the mixed unified manifest
would not be semantically correct.

For that reason, the new pycocotools-based evaluation is applied only to:

- `/home/lidz/YOLO/DataSet/Unified/manifests/coco2017_val.jsonl`

This is the cleanest way to obtain a more rigorous official-style metric
without pretending that VOC and COCO labels are already globally unified.

## 2. What Was Added

The following evaluation utilities were added:

- `/home/lidz/YOLO/yolov0/tools/evaluate.py`
  - internal engineering evaluation
  - reports:
    - `mAP@0.5`
    - `precision`
    - `recall`
    - `FLOPs`
    - `FPS`
- `/home/lidz/YOLO/yolov0/tools/evaluate_coco.py`
  - pycocotools-based COCO subset evaluation
  - reports:
    - `AP`
    - `AP50`
    - `AP75`
    - `AR@1`
    - `AR@10`
    - `AR@100`
    - `FLOPs`
    - `FPS`

Supporting files:

- `/home/lidz/YOLO/yolov0/utils/evaluation.py`
- `/home/lidz/YOLO/yolov0/utils/efficiency.py`
- `/home/lidz/YOLO/yolov0/utils/coco_eval.py`

## 3. Efficiency Metrics

The current efficiency metrics are:

- parameter count
- estimated FLOPs
- forward-only FPS

The FLOPs implementation is an in-project estimate based mainly on:

- `Conv2d`
- `Linear`

This is suitable for controlled internal comparison between our models.
It is not claimed to be a perfect official end-to-end system FLOPs metric.

The FPS benchmark is:

- forward-only
- fixed input size
- fixed batch size
- warmup + measured iterations

So it should be interpreted as:

> reproducible throughput under the current protocol

not as an all-conditions deployment FPS claim.

## 4. COCO Subset Official-Style Results

The following formal evaluation JSON files were produced:

- `/home/lidz/YOLO/yolov0/outputs/evaluations/deep_cnn_single_box_full_loss_coco_eval.json`
- `/home/lidz/YOLO/yolov0/outputs/evaluations/deep_residual_single_box_full_loss_coco_eval.json`
- `/home/lidz/YOLO/yolov0/outputs/evaluations/deep_residual_three_box_full_loss_coco_eval.json`
- `/home/lidz/YOLO/yolov0/outputs/evaluations/deep_residual_three_box_v5box_softobj_coco_eval.json`
- `/home/lidz/YOLO/yolov0/outputs/evaluations/deep_residual_three_box_v5box_softobj_assign_coco_eval.json`
- `/home/lidz/YOLO/yolov0/outputs/evaluations/deep_residual_three_box_v5box_softobj_clamp_coco_eval.json`
- `/home/lidz/YOLO/yolov0/outputs/evaluations/deep_residual_three_box_v5box_softobj_shapematch_coco_eval.json`

### 4.1 Result Table

| Model | Boxes per Cell | Params | FLOPs | FPS | COCO AP | COCO AP50 | COCO AP75 | COCO AR@100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `deep_cnn` + single-box | 1 | 17.66M | 31.56G | 140.50 | 0.000082 | 0.000486 | 0.000002 | 0.001081 |
| `deep_residual` + single-box | 1 | 26.52M | 42.18G | 113.28 | 0.000155 | 0.000588 | 0.000003 | 0.001211 |
| `deep_residual` + three-box | 3 | 26.56M | 42.19G | 116.44 | 0.000026 | 0.000122 | 0.000008 | 0.000999 |
| `deep_residual` + three-box + v5-box + soft-obj | 3 | 26.56M | 42.19G | 124.12 | 0.000069 | 0.000299 | 0.000006 | 0.001134 |
| `deep_residual` + three-box + v5-box + soft-obj + assign | 3 | 26.56M | 42.19G | 110.19 | 0.000056 | 0.000286 | 0.000002 | 0.001014 |
| `deep_residual` + three-box + v5-box + soft-obj + clamp | 3 | 26.56M | 42.19G | 116.71 | 0.000073 | 0.000296 | 0.000003 | 0.001090 |
| `deep_residual` + three-box + v5-box + soft-obj + shape-match | 3 | 26.56M | 42.19G | 122.42 | 0.000095 | 0.000343 | 0.000002 | 0.001332 |

## 5. Main Interpretation

### 5.1 Stage B is still the strongest practical system

Under the more official COCO-style metric:

- `deep_residual + single-box`

is still the best among the evaluated A/B/C models.

It beats:

- `deep_cnn + single-box`
- `deep_residual + three-box`

on:

- `AP`
- `AP50`
- `AR@100`

This agrees with the earlier conclusion drawn from validation loss.

### 5.2 Stage-C round 1 clearly improves the three-box line

The new Stage-C round-1 variant:

- uses YOLOv5-style box parameterization
- uses IoU-based soft objectness targets

Under the official COCO subset evaluation, this new three-box variant
improves substantially over the original stage-C model.

The clearest example is:

- `AP50` improves from about `0.000122`
- to about `0.000299`

So the first round of Stage-C fixes is meaningful and should be kept.

### 5.3 Stage-C round 2 does not improve the official COCO-style result

The new Stage-C round-2 variant:

- keeps the round-1 box parameterization and soft objectness target
- changes anchor assignment and ignore handling

This version improves the internal engineering metric:

- `mAP@0.5` rises from about `0.021857`
- to about `0.024256`

However, under the COCO subset evaluation:

- `AP50` falls slightly from about `0.000299`
- to about `0.000286`
- `AR@100` also falls slightly

So the round-2 assignment rewrite is:

- stable
- not useless
- but still not a clear quality win under the stricter metric

### 5.4 Round-3C is the strongest three-box variant so far

The new Stage-C round-3C variant:

- keeps the current anchors
- keeps YOLOv5-style box parameterization
- keeps IoU-based soft objectness with a floor
- replaces IoU-based matching with shape-ratio matching

This version is the first three-box branch that improves both:

- internal recall quality
- official-style COCO subset `AP50` / `AR@100`

Compared with the earlier three-box variants:

- `AP50` rises to about `0.000343`
- `AR@100` rises to about `0.001332`
- internal recall rises to about `0.201168`

So round-3C should be interpreted as:

- the strongest three-box variant so far
- strong evidence that matching standard, rather than just anchor count, is the core bottleneck

### 5.5 Stage C still underperforms the single-box residual baseline

Even after the round-3C fixes, the three-box system remains worse than the
single-box residual baseline on:

- internal `mAP@0.5`
- internal `precision`
- COCO `AP50`

At the same time, round-3C is now competitive in recall and even slightly
better on:

- COCO `AR@100`

This means the current three-box branch has started to unlock:

- better proposal coverage

but still has weaker ranking / scoring quality than the single-box residual
baseline.

### 5.6 Round-3B improves training stability more than final detection quality

The new Stage-C round-3B variant:

- keeps the current anchors
- keeps YOLOv5-style box parameterization
- keeps IoU-based soft objectness
- adds a positive objectness floor with:
  - `soft_objectness_min = 0.4`

This version shows a useful behavior change:

- the validation loss trajectory is smoother than round-2
- `COCO AP50` recovers from round-2 and returns close to round-1

However:

- internal `mAP@0.5` falls below round-2
- `COCO AP50` still does not beat round-1

So round-3B should be interpreted as:

- a valid stability-oriented fix
- not a decisive quality breakthrough

The current best practical branch is therefore still:

- `deep_residual + single-box full loss`

### 5.4 The current detector is still extremely weak in absolute terms

All COCO AP values are near zero.

This should not be hidden.
It means:

- the models are still early baselines
- the current head/loss/matching design is still far from mature
- structural and training improvements are still required before the
  detector can be considered strong in absolute detection quality

So the correct interpretation is:

> the metrics are useful for relative model comparison, but the current
> detector family is still weak in absolute detection performance

## 6. What This Means for the Next Stage

The official-style evaluation strengthens, rather than weakens, the
current direction choice.

At this moment:

- the best current practical branch is still:
  - `deep_residual + single-box full loss`
- the original three-box implementation is no longer the latest baseline
- the stage-C round-1, round-2, and round-3B variants are all accepted engineering
  branches
- but the three-box line still has not surpassed the
  single-box residual system on AP-oriented metrics

So the next design discussion should focus on:

- how to improve stage C further
- especially:
  - scoring quality after shape-matching
  - crowded-cell assignment
  - anchor usage rather than anchor numeric refit

The first two Stage-C round-1 fixes should now be treated as accepted:

- improved box parameterization
- IoU-based soft objectness

The Stage-C round-2 assignment rewrite should be treated as:

- a completed and formally evaluated branch
- informative for diagnosis
- but not the new best reference branch

The Stage-C round-3B objectness-floor change should be treated as:

- a completed and formally evaluated branch
- helpful for optimization stability
- insufficient on its own to make the three-box line surpass the best
  single-box residual baseline

The Stage-C round-3C shape-matching change should now be treated as:

- the strongest completed three-box branch so far
- a clear confirmation that matching standard is a central Stage-C bottleneck
- the new reference branch for any further three-box improvement

rather than immediately assuming that a three-box head is already better
just because it is structurally richer.

## 7. Status

This evaluation supplement is complete for the currently finished A/B/C
runs.

We now have:

- internal engineering evaluation
- official-style COCO subset evaluation
- efficiency metrics
- formal run artifacts

This is enough to support the next round of architecture and loss
discussion on a stronger empirical basis.
