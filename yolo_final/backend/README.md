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
export YOLO_BACKEND_CONFIG=configs/dual_scale_three_box_coco_only_noobj1_416.toml
export YOLO_BACKEND_CHECKPOINT=outputs/dual_scale_three_box_coco_only_noobj1_416_ddp_20260430_124818/best.pth
export YOLO_BACKEND_DEVICE=auto
export YOLO_BACKEND_ANNOTATION_DIR=backend/annotations
```

Future ONNX switch:

```bash
export YOLO_BACKEND_MODEL_FORMAT=onnx
export YOLO_BACKEND_ONNX_MODEL=exports/model.onnx
```

## Predict

```bash
curl -X POST http://127.0.0.1:5000/model_predict \
  -F image=@test.jpg \
  -F score_threshold=0.05 \
  -F top_k=100 \
  -F nms_iou_threshold=0.5
```

Response boxes are in original-image pixel `xyxy` coordinates.

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

This writes `backend/annotations/demo_001.txt` in YOLO format:

```text
0 0.171875 0.347222 0.156250 0.472222
```
