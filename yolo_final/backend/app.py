"""Flask REST API for model-assisted bbox annotation."""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from flask import Flask, jsonify, redirect, request, send_from_directory
from werkzeug.exceptions import RequestEntityTooLarge

from backend.annotation_io import save_uploaded_image, save_yolo_annotation
from backend.config import BackendSettings, load_settings
from backend.schemas import parse_float, parse_int, validate_annotation_payload
from utils.config import load_config


app = Flask(__name__)
settings = load_settings()
app.config["MAX_CONTENT_LENGTH"] = settings.max_upload_bytes
FRONTEND_DIR = PROJECT_ROOT / "frontend"
REACT_FRONTEND_DIST = PROJECT_ROOT / "frontend_react" / "dist"
_predictor = None
_predictor_error = None
_config_num_classes = None


def _json_error(message: str, status_code: int, **extra):
    payload = {"success": False, "error": message}
    payload.update(extra)
    return jsonify(payload), status_code


@app.errorhandler(RequestEntityTooLarge)
def handle_request_too_large(_exc):
    return _json_error(
        f"request body is too large; max upload size is {settings.max_upload_bytes} bytes",
        413,
    )


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
            use_fp16=current_settings.use_fp16,
        )
    if current_settings.model_format == "onnx":
        from backend.onnx_predictor import ONNXPredictor

        return ONNXPredictor(
            model_path=current_settings.onnx_model_path,
            config_path=current_settings.config_path,
            metadata_path=current_settings.metadata_path,
        )
    if current_settings.model_format == "torchscript":
        from backend.torchscript_predictor import TorchScriptPredictor

        return TorchScriptPredictor(
            model_path=current_settings.torchscript_model_path,
            config_path=current_settings.config_path,
            device_name=current_settings.device,
            metadata_path=current_settings.metadata_path,
            use_fp16=current_settings.use_fp16,
        )
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


def model_loaded() -> bool:
    return _predictor is not None


def warmup_predictor(predictor):
    """Run one tiny dummy prediction so the first user request avoids runtime warmup."""
    from PIL import Image

    image_size = int(getattr(predictor, "image_size", 416))
    dummy = Image.new("RGB", (image_size, image_size), color=(0, 0, 0))
    return predictor.predict(
        image=dummy,
        score_threshold=0.99,
        top_k=1,
        nms_iou_threshold=0.5,
    )


def get_config_num_classes() -> int:
    """Read num_classes from the active model config without forcing model load."""
    global _config_num_classes
    if _config_num_classes is None:
        _config_num_classes = int(load_config(settings.config_path)["data"]["num_classes"])
    return _config_num_classes


def validate_uploaded_image(image_file):
    """Open and validate one uploaded image, then rewind its stream."""
    if image_file is None:
        raise ValueError("missing multipart file field: image")
    try:
        from PIL import Image

        image = Image.open(image_file.stream).convert("RGB")
        image.load()
    except Exception as exc:
        raise ValueError("image must be a readable image file.") from exc
    finally:
        try:
            image_file.stream.seek(0)
        except Exception:
            pass

    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError("image width and height must be positive.")
    if width * height > settings.max_image_pixels:
        raise ValueError(
            f"image has too many pixels: {width * height} > {settings.max_image_pixels}"
        )
    return image


def parse_annotation_request() -> tuple[dict, object | None]:
    """Accept either JSON-only annotations or multipart image+annotation."""
    if request.content_type and request.content_type.startswith("multipart/form-data"):
        raw_annotation = request.form.get("annotation")
        if not raw_annotation:
            raise ValueError("multipart annotation request must include an `annotation` JSON field.")
        try:
            payload = json.loads(raw_annotation)
        except json.JSONDecodeError as exc:
            raise ValueError("annotation must be valid JSON.") from exc
        image_file = request.files.get("image")
        return validate_annotation_payload(payload), image_file
    return validate_annotation_payload(request.get_json(silent=True)), None


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
            "model_loaded": model_loaded(),
            "model_error": str(_predictor_error) if _predictor_error else "",
            "config": str(settings.config_path),
            "checkpoint": str(settings.checkpoint_path),
            "onnx_model": str(settings.onnx_model_path),
            "torchscript_model": str(settings.torchscript_model_path),
            "annotation_dir": str(settings.annotation_dir),
            "device": str(getattr(_predictor, "device", settings.device)),
            "num_classes": get_config_num_classes(),
            "limits": {
                "max_upload_bytes": settings.max_upload_bytes,
                "max_image_pixels": settings.max_image_pixels,
                "max_top_k": settings.max_top_k,
            },
        }
    )


@app.post("/model_warmup")
def model_warmup():
    """Load the model before the first real prediction request."""
    try:
        predictor = get_predictor()
        warmup_result = warmup_predictor(predictor)
        return jsonify(
            {
                "success": True,
                "model_loaded": True,
                "model": {
                    "format": settings.model_format,
                    "config": str(settings.config_path),
                    "checkpoint": str(settings.checkpoint_path),
                    "device": str(getattr(predictor, "device", settings.device)),
                },
                "latency_ms": warmup_result.get("latency_ms", {}),
            }
        )
    except Exception as exc:
        return _json_error(str(exc), 500, traceback=traceback.format_exc() if settings.debug else "")


@app.post("/model_predict")
def model_predict():
    """Receive one image and return model-predicted bbox proposals."""
    if "image" not in request.files:
        return _json_error("missing multipart file field: image", 400)

    try:
        score_threshold = parse_float(
            request.form.get("score_threshold"),
            default=0.5,
            name="score_threshold",
            minimum=0.0,
            maximum=1.0,
        )
        top_k = parse_int(
            request.form.get("top_k"),
            default=100,
            name="top_k",
            minimum=1,
            maximum=settings.max_top_k,
        )
        nms_iou_threshold = parse_float(
            request.form.get("nms_iou_threshold"),
            default=0.3,
            name="nms_iou_threshold",
            minimum=0.0,
            maximum=1.0,
        )
        image_file = request.files["image"]
        image = validate_uploaded_image(image_file)
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
        payload, image_file = parse_annotation_request()
        saved_image_path = ""
        if image_file is not None:
            validate_uploaded_image(image_file)
            saved_image_path = save_uploaded_image(
                image_file=image_file,
                image_id=payload["image_id"],
                image_dir=settings.annotation_dir / "images",
            )
        result = save_yolo_annotation(
            image_id=payload["image_id"],
            image_width=payload["image_width"],
            image_height=payload["image_height"],
            bboxes=payload["bboxes"],
            annotation_dir=settings.annotation_dir / "labels",
            num_classes=get_config_num_classes(),
        )
        return jsonify({"success": True, "saved_image_path": saved_image_path, **result})
    except ValueError as exc:
        return _json_error(str(exc), 400)
    except Exception as exc:
        return _json_error(str(exc), 500, traceback=traceback.format_exc() if settings.debug else "")


if __name__ == "__main__":
    if settings.preload_model:
        warmup_predictor(get_predictor())
    app.run(host=settings.host, port=settings.port, debug=settings.debug)
