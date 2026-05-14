# YOLO Final Backend

Flask REST backend for model-assisted bbox annotation and exported-model demos. The frontend API is stable: `/model_predict` accepts the same image and threshold fields for PyTorch `.pth`, TorchScript, and ONNX backends.

## Start

From `yolo_final/`:

```bash
scripts/start_yolo_detection_tool.sh
```

Static frontend:

```text
http://127.0.0.1:5000/
```

The default launcher starts the directly usable YOLO detection tool with the INT8
TorchScript CPU model. Other modes:

```bash
scripts/start_yolo_detection_tool.sh fp32-gpu
scripts/start_yolo_detection_tool.sh int8-cpu
scripts/start_yolo_detection_tool.sh pytorch
```

Built React frontend, after `npm run build` in `frontend_react/`:

```text
http://127.0.0.1:5000/react
```

Useful environment overrides:

```bash
export YOLO_BACKEND_MODEL_FORMAT=pytorch
export YOLO_BACKEND_CONFIG=configs/dual_scale_three_box_coco_only_noobj1_416_basic_aug_scale_jitter_50_lr7e4.toml
export YOLO_BACKEND_CHECKPOINT=outputs/dual_scale_three_box_coco_only_noobj1_416_basic_aug_scale_jitter_50_lr7e4_ddp_20260512_130823/best.pth
export YOLO_BACKEND_DEVICE=auto
export YOLO_BACKEND_ANNOTATION_DIR=backend/annotations
export YOLO_BACKEND_MAX_UPLOAD_MB=20
export YOLO_BACKEND_MAX_IMAGE_PIXELS=25000000
export YOLO_BACKEND_MAX_TOP_K=500
export YOLO_BACKEND_PRELOAD_MODEL=0
export YOLO_BACKEND_USE_FP16=1
```

TorchScript export switch:

```bash
export YOLO_BACKEND_MODEL_FORMAT=torchscript
export YOLO_BACKEND_CONFIG=configs/dual_scale_three_box_coco_only_noobj1_416_basic_aug_scale_jitter_50_lr7e4.toml
export YOLO_BACKEND_TORCHSCRIPT_MODEL=exports/checkpoint8/best_yolofinal_416_lr7e4.torchscript.pt
export YOLO_BACKEND_DEVICE=auto
python backend/app.py
```

ONNX switch:

```bash
export YOLO_BACKEND_MODEL_FORMAT=onnx
export YOLO_BACKEND_CONFIG=configs/dual_scale_three_box_coco_only_noobj1_416_basic_aug_scale_jitter_50_lr7e4.toml
export YOLO_BACKEND_ONNX_MODEL=exports/checkpoint8/best_yolofinal_416_lr7e4.onnx
python backend/app.py
```

ONNX requires `onnxruntime` at inference time. Export additionally requires `onnx`. In the current `yolov1` environment these packages are not installed, so the checkpoint 8 run records TorchScript as the available exported backend and ONNX as skipped by dependency status.

## Export And Benchmark

From `yolo_final/`:

```bash
python tools/export_model.py \
  --formats torchscript,onnx \
  --output-dir exports/checkpoint8 \
  --prefix best_yolofinal_416_lr7e4 \
  --device cuda \
  --batch-size 1
```

Benchmark the original PyTorch checkpoint and exported backend with the same Python post-processing:

```bash
python tools/benchmark_exported_model.py \
  --formats pytorch,torchscript,onnx \
  --torchscript-model exports/checkpoint8/best_yolofinal_416_lr7e4.torchscript.pt \
  --onnx-model exports/checkpoint8/best_yolofinal_416_lr7e4.onnx \
  --num-samples 64 \
  --warmup 8 \
  --vis-samples 16 \
  --score-threshold 0.5 \
  --top-k 10 \
  --nms-iou-threshold 0.5 \
  --output-json outputs/export_benchmark/checkpoint8_lr7e4_benchmark.json \
  --vis-dir outputs/export_benchmark/checkpoint8_lr7e4_vis16
```

Checkpoint 8 artifacts currently generated:

```text
exports/checkpoint8/best_yolofinal_416_lr7e4.torchscript.pt
exports/checkpoint8/best_yolofinal_416_lr7e4_int8_backbone_calib128.torchscript.pt
exports/checkpoint8/best_yolofinal_416_lr7e4.export_metadata.json
outputs/export_benchmark/checkpoint8_lr7e4_benchmark.json
outputs/export_benchmark/checkpoint8_lr7e4_int8_backbone_calib128_cpu_benchmark.json
outputs/export_benchmark/checkpoint8_lr7e4_vis16/
outputs/export_benchmark/checkpoint8_lr7e4_int8_backbone_calib128_cpu_vis16/
```

## INT8 PTQ

The detector can be quantized with CPU FX post-training static quantization. For stability,
the default keeps the detection head in FP32 and quantizes the backbone:

```bash
python tools/quantize_model.py \
  --calibration-samples 128 \
  --output-dir exports/checkpoint8 \
  --prefix best_yolofinal_416_lr7e4_int8_backbone_calib128 \
  --backend x86
```

CPU benchmark:

```bash
python tools/benchmark_exported_model.py \
  --formats pytorch,torchscript \
  --device cpu \
  --torchscript-model exports/checkpoint8/best_yolofinal_416_lr7e4_int8_backbone_calib128.torchscript.pt \
  --num-samples 64 \
  --warmup 8 \
  --vis-samples 16 \
  --score-threshold 0.5 \
  --top-k 10 \
  --nms-iou-threshold 0.5 \
  --output-json outputs/export_benchmark/checkpoint8_lr7e4_int8_backbone_calib128_cpu_benchmark.json \
  --vis-dir outputs/export_benchmark/checkpoint8_lr7e4_int8_backbone_calib128_cpu_vis16
```

COCO eval for the quantized TorchScript model:

```bash
python tools/evaluate_exported_coco.py \
  --device cpu \
  --output-json outputs/evaluations/checkpoint8_lr7e4_int8_backbone_calib128_coco_eval.json
```

Current result:

```text
COCO AP     0.191483
COCO AP50   0.309759
COCO AP75   0.199871
COCO AR100  0.351494
```

Use the quantized model in the Flask backend on CPU:

```bash
export YOLO_BACKEND_MODEL_FORMAT=torchscript
export YOLO_BACKEND_DEVICE=cpu
export YOLO_BACKEND_TORCHSCRIPT_MODEL=exports/checkpoint8/best_yolofinal_416_lr7e4_int8_backbone_calib128.torchscript.pt
python backend/app.py
```

## Health And Warmup

```bash
curl http://127.0.0.1:5000/health
curl -X POST http://127.0.0.1:5000/model_warmup
```

`/health` reports whether the model is loaded, the active config/checkpoint, class count,
runtime device, and request limits. `/model_warmup` loads the model before the first real
prediction request, which keeps the first UI prediction from paying the model load cost.

## Predict

```bash
curl -X POST http://127.0.0.1:5000/model_predict \
  -F image=@test.jpg \
  -F score_threshold=0.05 \
  -F top_k=100 \
  -F nms_iou_threshold=0.5
```

Response boxes are in original-image pixel `xyxy` coordinates.
The response also includes `latency_ms` with preprocess, inference, postprocess, and total time.

## Save Human Annotation

```bash
curl -X POST http://127.0.0.1:5000/human_annotate \
  -H "Content-Type: application/json" \
  -d '{
    "image_id": "demo_001",
    "image_width": 1280,
    "image_height": 720,
    "bboxes": [
      {"class_id": 0, "x1": 120, "y1": 80, "x2": 320, "y2": 420}
    ]
  }'
```

This writes `backend/annotations/labels/demo_001.txt` in YOLO format:

```text
0 0.171875 0.347222 0.156250 0.472222
```

To save the uploaded image and annotation together:

```bash
curl -X POST http://127.0.0.1:5000/human_annotate \
  -F image=@test.jpg \
  -F 'annotation={
    "image_id": "demo_001",
    "image_width": 1280,
    "image_height": 720,
    "bboxes": [
      {"class_id": 0, "x1": 120, "y1": 80, "x2": 320, "y2": 420}
    ]
  }'
```
