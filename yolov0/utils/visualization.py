"""Visualization helpers for saving GT-versus-prediction detector images."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image, ImageDraw

from utils.prediction import decode_predictions_for_image, select_prediction_for_image


def tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    """Convert one CHW float image tensor in `[0, 1]` into a PIL image."""
    image = (image_tensor.clamp(0.0, 1.0) * 255.0).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(image)


def draw_gt_and_predictions(
    image: Image.Image,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    predictions: list[dict],
) -> Image.Image:
    """Draw green GT boxes and red prediction boxes on one image."""
    draw = ImageDraw.Draw(image)

    for box, label in zip(gt_boxes.tolist(), gt_labels.tolist()):
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline="lime", width=2)
        draw.text((x1 + 2, y1 + 2), f"gt:{label}", fill="lime")

    for pred in predictions:
        x1, y1, x2, y2 = pred["box_xyxy"]
        class_id = pred["class_id"]
        score = pred["score"]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1 + 2, max(y1 - 12, 0)), f"pred:{class_id} {score:.2f}", fill="red")

    return image


def save_visualization_set(
    model,
    dataset,
    output_dir: Path,
    device,
    num_classes: int,
    num_boxes: int = 1,
    anchors: list[tuple[float, float]] | None = None,
    box_parameterization: str = "legacy",
    max_samples: int = 4,
    score_threshold: float = 0.05,
    top_k: int = 10,
    score_alpha: float = 1.0,
    score_beta: float = 1.0,
):
    """Run prediction on a fixed dataset prefix and save GT-vs-pred visualization images."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    num_samples = min(max_samples, len(dataset))
    with torch.no_grad():
        for sample_index in range(num_samples):
            image_tensor, target = dataset[sample_index]
            pred = select_prediction_for_image(model(image_tensor.unsqueeze(0).to(device)), 0)
            predictions = decode_predictions_for_image(
                pred=pred,
                image_size=int(target["resized_size"][0].item()),
                num_classes=num_classes,
                num_boxes=num_boxes,
                anchors=anchors,
                box_parameterization=box_parameterization,
                score_threshold=score_threshold,
                top_k=top_k,
                score_alpha=score_alpha,
                score_beta=score_beta,
            )
            image = tensor_to_pil(image_tensor)
            image = draw_gt_and_predictions(image, target["boxes"], target["labels"], predictions)
            sample_id = target["sample_id"]
            image.save(output_dir / f"{sample_id}.png")
