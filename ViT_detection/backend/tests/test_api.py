import io
import json
from dataclasses import replace
from pathlib import Path

from PIL import Image

import backend.app as backend_app


def make_png_bytes() -> bytes:
    image = Image.new("RGB", (32, 24), color=(20, 80, 120))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def configure_tmp_app(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(backend_app, "settings", replace(backend_app.settings, annotation_dir=tmp_path))
    monkeypatch.setattr(backend_app, "_config_num_classes", 80)
    backend_app.app.config["TESTING"] = True
    backend_app.app.config["MAX_CONTENT_LENGTH"] = backend_app.settings.max_upload_bytes
    return backend_app.app.test_client()


def test_health_reports_limits_and_model_state(tmp_path, monkeypatch):
    client = configure_tmp_app(tmp_path, monkeypatch)
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["success"] is True
    assert payload["model_loaded"] is False
    assert payload["limits"]["max_top_k"] >= 1


def test_human_annotate_json_writes_label(tmp_path, monkeypatch):
    client = configure_tmp_app(tmp_path, monkeypatch)
    response = client.post(
        "/human_annotate",
        json={
            "image_id": "sample_001.jpg",
            "image_width": 100,
            "image_height": 50,
            "bboxes": [{"class_id": 0, "x1": 10, "y1": 5, "x2": 30, "y2": 25}],
        },
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["success"] is True
    assert Path(payload["saved_path"]).read_text(encoding="utf-8").strip() == (
        "0 0.200000 0.300000 0.200000 0.400000"
    )
    assert Path(payload["saved_path"]).parent.name == "labels"


def test_human_annotate_multipart_saves_image_and_label(tmp_path, monkeypatch):
    client = configure_tmp_app(tmp_path, monkeypatch)
    annotation = {
        "image_id": "uploaded.png",
        "image_width": 32,
        "image_height": 24,
        "bboxes": [{"class_id": 1, "x1": 4, "y1": 3, "x2": 20, "y2": 15}],
    }
    response = client.post(
        "/human_annotate",
        data={
            "annotation": json.dumps(annotation),
            "image": (io.BytesIO(make_png_bytes()), "uploaded.png"),
        },
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert Path(payload["saved_image_path"]).exists()
    assert Path(payload["saved_image_path"]).parent.name == "images"
    assert Path(payload["saved_path"]).exists()


def test_model_predict_rejects_top_k_above_limit(tmp_path, monkeypatch):
    client = configure_tmp_app(tmp_path, monkeypatch)
    response = client.post(
        "/model_predict",
        data={
            "top_k": str(backend_app.settings.max_top_k + 1),
            "image": (io.BytesIO(make_png_bytes()), "image.png"),
        },
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert "top_k" in response.get_json()["error"]
