from pathlib import Path

import pytest

from backend.annotation_io import save_yolo_annotation, sanitize_image_id, xyxy_to_yolo


def test_xyxy_to_yolo_normalizes_coordinates():
    class_id, center_x, center_y, width, height = xyxy_to_yolo(
        {"class_id": 2, "x1": 10, "y1": 5, "x2": 30, "y2": 25},
        image_width=100,
        image_height=50,
    )
    assert class_id == 2
    assert center_x == pytest.approx(0.2)
    assert center_y == pytest.approx(0.3)
    assert width == pytest.approx(0.2)
    assert height == pytest.approx(0.4)


def test_save_yolo_annotation_clips_and_writes(tmp_path: Path):
    result = save_yolo_annotation(
        image_id="../demo image.jpg",
        image_width=100,
        image_height=50,
        bboxes=[{"class_id": 1, "x1": -10, "y1": -5, "x2": 30, "y2": 25}],
        annotation_dir=tmp_path,
        num_classes=3,
    )
    saved_path = Path(result["saved_path"])
    assert saved_path.name == "demo_image.txt"
    assert saved_path.read_text(encoding="utf-8").strip() == "1 0.150000 0.250000 0.300000 0.500000"
    assert result["bboxes"][0]["x1"] == 0.0


def test_save_yolo_annotation_rejects_out_of_range_class(tmp_path: Path):
    with pytest.raises(ValueError, match="class_id must be < num_classes"):
        save_yolo_annotation(
            image_id="demo",
            image_width=100,
            image_height=50,
            bboxes=[{"class_id": 3, "x1": 1, "y1": 1, "x2": 10, "y2": 10}],
            annotation_dir=tmp_path,
            num_classes=3,
        )


def test_sanitize_image_id_rejects_empty_values():
    with pytest.raises(ValueError):
        sanitize_image_id("...")

