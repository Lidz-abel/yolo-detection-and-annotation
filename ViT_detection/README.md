# ViT_detection

`ViT_detection` is a ViT-backbone version of the existing `yolo_final` detector.
It intentionally preserves the old training, loss, evaluation, export, and
backend interfaces while replacing the CNN feature extractor with a
detection-oriented Vision Transformer.

## Structure

```text
ViT_detection/
├── backend/
├── configs/
├── data/
├── engine/
├── exports/
├── frontend/
├── frontend_react/
├── logs/
├── losses/
├── models/
├── outputs/
├── report/
├── runs/
├── scripts/
├── tools/
└── utils/
```

## Architecture

See `ARCHITECTURE.md`.

Current improved version:

```text
416x416 image
-> overlapping patch stem
-> p3 52x52 feature map
-> p4 26x26 token grid
-> Transformer encoder
-> p4 26x26 feature map
-> stride-2 downsample
-> p5 13x13 feature map
-> FPN/PAN lite neck
-> decoupled YOLO head
-> YOLO-style loss and prediction decode
```

## Smoke Check

```bash
cd /home/lidz/YOLO/ViT_detection
python tools/smoke_dual_scale_three_box.py --config configs/hybrid_vit_p3p4p5_416_ddp.toml
```

## DDP Train

```bash
cd /home/lidz/YOLO/ViT_detection
NPROC_PER_NODE=8 scripts/run_hybrid_vit_p3p4p5_416_ddp.sh
```

The first improved config keeps moderate bbox-safe augmentation and uses
`tools/train_ddp.py` through `torchrun`. It defaults to a global batch size of
64 and a lower learning rate than the CNN run because attention over a 26x26
token grid is heavier and less forgiving from scratch.

The DDP training path also enables AMP, gradient clipping, EMA validation
weights, cosine warmup, and SyncBatchNorm. `best.pth` is saved from the EMA
weights when EMA is enabled; `last.pth` keeps the raw training weights, and
`last_ema.pth` keeps the smoothed weights.

## First-Run Recipe

The default config is now tuned for getting a usable first training run rather
than proving a from-scratch comparison:

- `neck` and `head` are partially initialized from the mature
  `yolo_final` p3/p4/p5 COCO checkpoint.
- The Hybrid ViT channel layout is aligned to that checkpoint:
  `p3=256`, `p4=256`, `p5=512`.
- `neck` and `head` are frozen for the first 2 epochs so the new backbone can
  adapt to a trained detector head before all modules are fine-tuned together.
- Detection layers use a lower learning-rate multiplier after unfreezing, while
  the randomly initialized backbone receives the main learning rate.
- Augmentation is kept, but made conservative enough for the first run to
  stabilize quickly.
