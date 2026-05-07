"""Helpers for validating boxes and saving YOLO-format annotation files."""

from __future__ import annotations

import re
from pathlib import Path


_SAFE_STEM_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")


def sanitize_image_id(image_id: str) -> str:
    """Return a filesystem-safe image id stem."""
    text = str(image_id or "").strip()
    if not text:
        raise ValueError("image_id must be a non-empty string.")
    stem = Path(text).stem
    stem = _SAFE_STEM_PATTERN.sub("_", stem).strip("._")
    if not stem:
        raise ValueError("image_id does not contain a valid filename stem.")
    return stem


def clip_xyxy(box: dict, image_width: int, image_height: int) -> dict:
    """Clip one xyxy box to image bounds and validate that it has area."""
    try:
        class_id = int(box["class_id"])
        x1 = float(box["x1"])
        y1 = float(box["y1"])
        x2 = float(box["x2"])
        y2 = float(box["y2"])
    except KeyError as exc:
        raise ValueError(f"bbox is missing required field: {exc.args[0]}") from exc
    except (TypeError, ValueError) as exc:
        raise ValueError("bbox fields must be numeric.") from exc

    if class_id < 0:
        raise ValueError("class_id must be non-negative.")

    width = float(image_width)
    height = float(image_height)
    x1 = max(0.0, min(x1, width))
    x2 = max(0.0, min(x2, width))
    y1 = max(0.0, min(y1, height))
    y2 = max(0.0, min(y2, height))

    if x2 <= x1 or y2 <= y1:
        raise ValueError("bbox must satisfy x2 > x1 and y2 > y1 after clipping.")

    return {
        "class_id": class_id,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
    }


def xyxy_to_yolo(box: dict, image_width: int, image_height: int) -> tuple[int, float, float, float, float]:
    """Convert clipped xyxy pixel coordinates to normalized YOLO cxcywh."""
    clipped = clip_xyxy(box, image_width=image_width, image_height=image_height)
    width = float(image_width)
    height = float(image_height)
    box_width = clipped["x2"] - clipped["x1"]
    box_height = clipped["y2"] - clipped["y1"]
    center_x = clipped["x1"] + box_width * 0.5
    center_y = clipped["y1"] + box_height * 0.5
    return (
        clipped["class_id"],
        center_x / width,
        center_y / height,
        box_width / width,
        box_height / height,
    )


def save_yolo_annotation(
    image_id: str,
    image_width: int,
    image_height: int,
    bboxes: list[dict],
    annotation_dir: Path,
) -> dict:
    """Validate boxes and save one YOLO txt annotation file."""
    if int(image_width) <= 0 or int(image_height) <= 0:
        raise ValueError("image_width and image_height must be positive.")
    if not isinstance(bboxes, list):
        raise ValueError("bboxes must be a list.")

    safe_stem = sanitize_image_id(image_id)
    annotation_dir.mkdir(parents=True, exist_ok=True)
    output_path = annotation_dir / f"{safe_stem}.txt"

    lines = []
    clipped_boxes = []
    for box in bboxes:
        clipped = clip_xyxy(box, image_width=int(image_width), image_height=int(image_height))
        clipped_boxes.append(clipped)
        class_id, center_x, center_y, width, height = xyxy_to_yolo(
            clipped,
            image_width=int(image_width),
            image_height=int(image_height),
        )
        lines.append(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")

    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return {
        "saved_path": str(output_path),
        "num_boxes": len(lines),
        "bboxes": clipped_boxes,
    }

