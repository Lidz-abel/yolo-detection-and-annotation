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
- `/home/lidz/YOLO/yolov0/outputs/evaluations/deep_residual_three_box_v5box_softobj_shapematch_tight_coco_eval.json`
- `/home/lidz/YOLO/yolov0/outputs/evaluations/deep_residual_three_box_v5box_softobj_shapematch_ignore_coco_eval.json`
- `/home/lidz/YOLO/yolov0/outputs/evaluations/deep_residual_three_box_qualitycls_coco_eval.json`
- `/home/lidz/YOLO/yolov0/outputs/evaluations/deep_residual_three_box_qualitycls_decoupled_coco_eval.json`
- `/home/lidz/YOLO/yolov0/outputs/evaluations/deep_residual_three_box_dynamicassign_coco_eval.json`
- `/home/lidz/YOLO/yolov0/outputs/evaluations/deep_residual_three_box_dynamicassign_scoretune_coco_eval.json`
- `/home/lidz/YOLO/yolov0/outputs/evaluations/deep_residual_three_box_dynamicassign_topk1_coco_eval.json`

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
| `deep_residual` + three-box + v5-box + soft-obj + shape-match + tight | 3 | 26.56M | 42.19G | 120.57 | 0.000159 | 0.000496 | 0.000126 | 0.001103 |
| `deep_residual` + three-box + v5-box + soft-obj + shape-match + ignore | 3 | 26.56M | 42.19G | 120.01 | 0.000135 | 0.000432 | 0.000022 | 0.001175 |
| `deep_residual` + three-box + quality-cls | 3 | 26.56M | 42.19G | 100.69 | 0.000097 | 0.000415 | 0.000007 | 0.000782 |
| `deep_residual` + three-box + quality-cls + decoupled | 3 | 28.92M | 42.66G | 93.81 | 0.000206 | 0.000814 | 0.000021 | 0.002500 |
| `deep_residual` + three-box + dynamic-assign | 3 | 28.92M | 42.66G | 98.41 | 0.000299 | 0.001150 | 0.000120 | 0.003097 |
| `deep_residual` + three-box + dynamic-assign + topk1 | 3 | 28.92M | 42.66G | 102.39 | 0.000175 | 0.000854 | 0.000011 | 0.003093 |
| `deep_residual` + three-box + dynamic-assign + score-tune | 3 | 28.92M | 42.66G | 90.26 | 0.000348 | 0.001336 | 0.000154 | 0.002482 |
| `deep_residual` + three-box + dynamic-assign + topk1 | 3 | 28.92M | 42.66G | 106.60 | 0.000175 | 0.000854 | 0.000011 | 0.003093 |

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

### 5.5 Round-5C pushes the three-box line beyond the Round-5B baseline

The new Round-5C variant:

- keeps the Round-5B decoupled head
- keeps quality-aware classification
- keeps the current anchors and YOLOv5-style box parameterization
- replaces the static positive assignment with a dynamic cost-based assignment

Its COCO subset metrics are:

- `AP50 = 0.001150`
- `AR@100 = 0.003097`

Its internal engineering metrics are:

- `mAP@0.5 = 0.086128`
- `precision = 0.048140`
- `recall = 0.306746`

This means Round 5C becomes the new strongest three-box branch so far.
Its main tradeoff is not stability, but prediction volume:

- `num_predictions = 334960`

So the next bottleneck is no longer whether the model can cover targets,
but how to keep the improved recall while further improving ranking quality
and suppressing surplus boxes.

### 5.6 Round-6A improves AP50 by retuning the final ranking formula

Round 6A does not retrain the detector. It keeps the Round 5C weights fixed
and only changes the final score to:

- `score = obj^2 * cls`

The result is:

- internal `precision` rises from `0.048140` to `0.084336`
- internal `num_predictions` falls from `334960` to `174813`
- COCO `AP50` rises from `0.001150` to `0.001336`

The cost is:

- internal `recall` falls from `0.306746` to `0.280456`
- COCO `AR@100` falls from `0.003097` to `0.002482`

So Round 6A is a successful inference-side ranking improvement, but not a
new training baseline by itself.

### 5.7 Round-6C over-tightens dynamic assignment

Round 6C keeps the Round 5C detector design but reduces:

- `dynamic_topk: 2 -> 1`

The expectation was to reduce surplus positive slots and improve ranking.
The actual result is:

- internal `mAP@0.5` drops to `0.057019`
- internal `precision` drops to `0.031924`
- internal `recall` drops to `0.270526`
- `num_predictions` rises to `445458`
- COCO `AP50` falls to `0.000854`
- COCO `AR@100` stays near `0.003093`

This means Round 6C does **not** become the new mainline. It shows that
over-tightening dynamic assignment damages the precision/recall balance
without giving a cleaner prediction set.

The practical takeaway is:

- keep Round 5C as the strongest training-side three-box baseline
- keep Round 6A as the strongest inference-side score-tuning result
- do not carry Round 6C forward as the base for the next experiment

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
### 5.7 Round-5A confirms that cls scoring is one of the current bottlenecks

The new Stage-C round-5A variant:

- keeps the current `shape-matching + ignore band`
- keeps the current YOLOv5-style box parameterization
- keeps IoU-based soft objectness with a floor
- replaces hard one-hot positive cls targets with IoU-aware soft cls targets

This version shows a very specific tradeoff:

- internal precision rises sharply
- total predicted boxes drop substantially
- but recall and `AR@100` fall noticeably

Under the stricter metrics:

- `AP50` is below round-4A
- `AR@100` is below round-4B

So round-5A should be interpreted as:

- strong evidence that three-box quality ranking is still a real bottleneck
- evidence that quality-aware classification can suppress low-quality duplicate boxes

### 5.8 Round-6A improves AP50 by retuning the final ranking formula

The new Round-6A variant:

- keeps the Round-5C checkpoint fixed
- does not retrain the detector
- changes the final score to:
  - `score = obj^2.0 * cls^1.0`

Its internal engineering metrics are:

- `mAP@0.5 = 0.085815`
- `precision = 0.084336`
- `recall = 0.280456`
- `num_predictions = 174813`

Its COCO subset metrics are:

- `AP50 = 0.001336`
- `AR@100 = 0.002482`

Compared with Round 5C, this means:

- the number of predicted boxes drops sharply
- precision rises substantially
- AP50 rises further
- recall and AR@100 fall moderately

So Round 6A should be interpreted as:

- strong evidence that the current three-box bottleneck is not only assignment
- direct evidence that better ranking alone can still lift AP50
- a signal that future training-side changes should continue to align `obj`
  and `cls` with box quality
- but also evidence that this first implementation is too aggressive and over-suppresses borderline positives

In other words, round-5A is valuable mainly because it confirms the direction
of the next step:

- decoupling the classification branch from the regression/objectness branch

instead of keeping all three tasks tied to the same head representation.

### 5.8 Round-5B is the strongest Stage-C result so far

The new Stage-C round-5B variant:

- keeps the round-5A quality-aware cls target
- keeps the round-4B `shape-matching + ignore band`
- replaces the shared detection head with a decoupled head

This version is the first Stage-C branch that clearly improves both:

- ranking quality
- proposal coverage

relative to the earlier three-box variants.

The strongest signals are:

- internal `mAP@0.5` jumps to about `0.0779`
- internal recall rises to about `0.2596`
- COCO `AP50` rises to about `0.000814`
- COCO `AR@100` rises to about `0.002500`

It also exceeds the previous practical baseline:

- `deep_residual + single-box`

on both:

- COCO `AP50`
- COCO `AR@100`

So round-5B should be interpreted as:

- evidence that head coupling was one of the main remaining Stage-C bottlenecks
- the first three-box branch with clear practical value under the current checkpoint-6 protocol
- the new baseline for any further Stage-C matcher work
- not a decisive quality breakthrough

### 5.7 Round-4A improves AP50 by tightening the three-box mechanism

The new Stage-C round-4A variant:

- tightens `anchor_shape_ratio` from `4.0` to `2.5`
- lowers `soft_objectness_min` from `0.4` to `0.05`
- keeps the current `shape-matching` assignment framework unchanged

This variant does **not** improve every metric, but it changes the three-box
behavior in a useful direction:

- the number of predictions drops
- internal precision rises slightly
- COCO `AP50` rises clearly

At the same time:

- recall drops slightly
- `AR@100` also falls slightly

So round-4A should be interpreted as:

- a precision-oriented tightening step
- the first Stage-C variant that noticeably reduces overprediction while
  improving official-style `AP50`

### 5.8 Round-4B activates a real ignore band and shifts the tradeoff back toward recall

The new Stage-C round-4B variant:

- keeps the round-4A tight `shape_ratio` setting
- introduces `anchor_ignore_shape_ratio = 4.0`
- activates a real `positive / ignore / negative` split in the
  `shape-matching` branch

This version is important mainly because the ignore mechanism is now truly
working:

- `ignored_count` becomes clearly non-zero in both train and val

Under the internal and COCO-style metrics, round-4B behaves like a recall-
oriented counterpart to round-4A:

- internal `mAP@0.5` rises back above round-4A
- internal recall rises
- COCO `AR@100` rises above round-4A

but:

- internal precision falls
- COCO `AP50` falls from the round-4A peak
- prediction count increases again

So round-4B should be interpreted as:

- a successful ignore-band activation
- evidence that Stage C is now trading between coverage and ranking quality
- not yet the final best three-box variant if `AP50` is the main priority

### 5.7 Round-4A improves AP50 by tightening the three-box system

The new Stage-C round-4A variant:

- keeps `shape-matching`
- tightens `anchor_shape_ratio` from `4.0` to `2.5`
- lowers `soft_objectness_min` from `0.4` to `0.05`
- raises the visualization threshold for cleaner GT-vs-Pred inspection

This round changes no backbone code and does not touch the core loss
structure. It is a pure tightening experiment.

Under evaluation, round-4A shows a very specific trade-off:

- internal `num_predictions` drops from about `301567`
  to about `293968`
- internal `precision` rises slightly
- internal `mAP@0.5` falls slightly
- COCO `AP50` rises sharply from about `0.000343`
  to about `0.000496`
- COCO `AR@100` falls from about `0.001332`
  to about `0.001103`

So round-4A should be interpreted as:

- a successful reduction of over-prediction
- a clear improvement in ranking / precision quality
- a partial sacrifice of recall

This is important because it confirms that the current Stage-C problem is
no longer only “can the three-box head cover enough objects?” but also
“can it stop rewarding too many low-quality boxes?”

The current best practical branch is therefore still:

- `deep_residual + single-box full loss`

### 5.8 The current detector is still extremely weak in absolute terms

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
