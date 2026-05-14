"""Small request parsing helpers for the Flask backend."""

from __future__ import annotations


def parse_float(value, default: float, name: str, minimum: float | None = None, maximum: float | None = None) -> float:
    """Parse one optional float request parameter."""
    if value is None or value == "":
        result = float(default)
    else:
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a float.") from exc
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be <= {maximum}.")
    return result


def parse_int(value, default: int, name: str, minimum: int | None = None, maximum: int | None = None) -> int:
    """Parse one optional integer request parameter."""
    if value is None or value == "":
        result = int(default)
    else:
        try:
            result = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be an integer.") from exc
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be <= {maximum}.")
    return result


def validate_annotation_payload(payload: dict) -> dict:
    """Validate the JSON body accepted by /human_annotate."""
    if not isinstance(payload, dict):
        raise ValueError("JSON body must be an object.")
    for key in ("image_id", "image_width", "image_height", "bboxes"):
        if key not in payload:
            raise ValueError(f"missing required field: {key}")
    try:
        image_width = int(payload["image_width"])
        image_height = int(payload["image_height"])
    except (TypeError, ValueError) as exc:
        raise ValueError("image_width and image_height must be integers.") from exc
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image_width and image_height must be positive.")
    if not isinstance(payload["bboxes"], list):
        raise ValueError("bboxes must be a list.")
    return {
        "image_id": str(payload["image_id"]),
        "image_width": image_width,
        "image_height": image_height,
        "bboxes": payload["bboxes"],
    }

