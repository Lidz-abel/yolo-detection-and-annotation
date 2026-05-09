# YOLO Final Backend

Flask REST backend for checkpoint 9.2. The API is stable for the frontend: the current implementation uses the PyTorch `.pth` checkpoint, while the ONNX predictor slot is reserved for checkpoint 8 exports.

## Start

From `yolo_final/`:

```bash
pip install -r backend/requirements.txt
python backend/app.py
```

Static frontend:

```text
http://127.0.0.1:5000/
```

Built React frontend, after `npm run build` in `frontend_react/`:

```text
http://127.0.0.1:5000/react
```

Useful environment overrides:

```bash
export YOLO_BACKEND_MODEL_FORMAT=pytorch
export YOLO_BACKEND_CONFIG=configs/dual_scale_three_box_coco_only_noobj1_416_basic_aug.toml
export YOLO_BACKEND_CHECKPOINT=outputs/dual_scale_three_box_coco_only_noobj1_416_basic_aug_ddp_20260508_014013/best.pth
export YOLO_BACKEND_DEVICE=auto
export YOLO_BACKEND_ANNOTATION_DIR=backend/annotations
export YOLO_BACKEND_MAX_UPLOAD_MB=20
export YOLO_BACKEND_MAX_IMAGE_PIXELS=25000000
export YOLO_BACKEND_MAX_TOP_K=500
export YOLO_BACKEND_PRELOAD_MODEL=0
export YOLO_BACKEND_USE_FP16=1
```

Future ONNX switch:

```bash
export YOLO_BACKEND_MODEL_FORMAT=onnx
export YOLO_BACKEND_ONNX_MODEL=exports/model.onnx
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
