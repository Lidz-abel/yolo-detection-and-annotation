"""Flask REST API for model-assisted bbox annotation."""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from flask import Flask, jsonify, redirect, request, send_from_directory

from backend.annotation_io import save_yolo_annotation
from backend.config import BackendSettings, load_settings
from backend.schemas import parse_float, parse_int, validate_annotation_payload


app = Flask(__name__)
settings = load_settings()
FRONTEND_DIR = PROJECT_ROOT / "frontend"
REACT_FRONTEND_DIST = PROJECT_ROOT / "frontend_react" / "dist"
_predictor = None
_predictor_error = None


def _json_error(message: str, status_code: int, **extra):
    payload = {"success": False, "error": message}
    payload.update(extra)
    return jsonify(payload), status_code


@app.after_request
def add_cors_headers(response):
    """Allow the standalone frontend to call the API during local demos."""
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response


def _build_predictor(current_settings: BackendSettings):
    if current_settings.model_format == "pytorch":
        from backend.pytorch_predictor import PyTorchPredictor

        return PyTorchPredictor(
            config_path=current_settings.config_path,
            checkpoint_path=current_settings.checkpoint_path,
            device_name=current_settings.device,
            metadata_path=current_settings.metadata_path,
        )
    if current_settings.model_format == "onnx":
        from backend.onnx_predictor import ONNXPredictor

        return ONNXPredictor(current_settings.onnx_model_path)
    raise ValueError(f"Unsupported YOLO_BACKEND_MODEL_FORMAT: {current_settings.model_format}")


def get_predictor():
    """Lazily load the configured predictor for /model_predict."""
    global _predictor, _predictor_error
    if _predictor is not None:
        return _predictor
    if _predictor_error is not None:
        raise _predictor_error
    try:
        _predictor = _build_predictor(settings)
        return _predictor
    except Exception as exc:
        _predictor_error = exc
        raise


@app.get("/")
def frontend_index():
    """Serve the annotation frontend from the Flask backend."""
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.get("/frontend/<path:filename>")
def frontend_assets(filename):
    """Serve frontend static assets."""
    return send_from_directory(FRONTEND_DIR, filename)


@app.get("/react/")
def react_frontend_index():
    """Serve the built React annotation frontend when available."""
    if not (REACT_FRONTEND_DIST / "index.html").exists():
        return _json_error(
            "React frontend has not been built. Run `npm install` and `npm run build` in frontend_react, "
            "or use the Vite dev server.",
            404,
        )
    return send_from_directory(REACT_FRONTEND_DIST, "index.html")


@app.get("/react")
def react_frontend_redirect():
    """Normalize the React app URL so relative Vite assets resolve correctly."""
    return redirect("/react/", code=308)


@app.get("/react/<path:filename>")
def react_frontend_assets(filename):
    """Serve built React frontend assets."""
    return send_from_directory(REACT_FRONTEND_DIST, filename)


@app.get("/health")
def health():
    return jsonify(
        {
            "success": True,
            "model_format": settings.model_format,
            "config": str(settings.config_path),
            "checkpoint": str(settings.checkpoint_path),
            "onnx_model": str(settings.onnx_model_path),
            "annotation_dir": str(settings.annotation_dir),
        }
    )


@app.post("/model_predict")
def model_predict():
    """Receive one image and return model-predicted bbox proposals."""
    if "image" not in request.files:
        return _json_error("missing multipart file field: image", 400)

    try:
        from PIL import Image

        score_threshold = parse_float(
            request.form.get("score_threshold"),
            default=0.05,
            name="score_threshold",
            minimum=0.0,
            maximum=1.0,
        )
        top_k = parse_int(request.form.get("top_k"), default=100, name="top_k", minimum=1)
        nms_iou_threshold = parse_float(
            request.form.get("nms_iou_threshold"),
            default=0.5,
            name="nms_iou_threshold",
            minimum=0.0,
            maximum=1.0,
        )
        image_file = request.files["image"]
        image = Image.open(image_file.stream).convert("RGB")
        predictor = get_predictor()
        result = predictor.predict(
            image=image,
            score_threshold=score_threshold,
            top_k=top_k,
            nms_iou_threshold=nms_iou_threshold,
        )
        return jsonify(
            {
                "success": True,
                "image_id": image_file.filename or "uploaded_image",
                "score_threshold": score_threshold,
                "top_k": top_k,
                "nms_iou_threshold": nms_iou_threshold,
                **result,
            }
        )
    except NotImplementedError as exc:
        return _json_error(str(exc), 501)
    except ValueError as exc:
        return _json_error(str(exc), 400)
    except Exception as exc:
        return _json_error(str(exc), 500, traceback=traceback.format_exc() if settings.debug else "")


@app.post("/human_annotate")
def human_annotate():
    """Receive edited boxes and save a YOLO-format annotation txt file."""
    try:
        payload = validate_annotation_payload(request.get_json(silent=True))
        result = save_yolo_annotation(
            image_id=payload["image_id"],
            image_width=payload["image_width"],
            image_height=payload["image_height"],
            bboxes=payload["bboxes"],
            annotation_dir=settings.annotation_dir,
        )
        return jsonify({"success": True, **result})
    except ValueError as exc:
        return _json_error(str(exc), 400)
    except Exception as exc:
        return _json_error(str(exc), 500, traceback=traceback.format_exc() if settings.debug else "")


if __name__ == "__main__":
    app.run(host=settings.host, port=settings.port, debug=settings.debug)
