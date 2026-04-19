from pathlib import Path

import torch
from PIL import Image, ImageDraw

from utils.prediction import decode_predictions


def tensor_to_pil(image_tensor):
    image = (image_tensor.clamp(0.0, 1.0) * 255.0).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(image)


def draw_boxes(image, gt_boxes, gt_labels, pred_boxes):
    draw = ImageDraw.Draw(image)

    for box, label in zip(gt_boxes, gt_labels):
        x1, y1, x2, y2 = [float(v) for v in box]
        draw.rectangle([x1, y1, x2, y2], outline="lime", width=2)
        draw.text((x1 + 2, y1 + 2), f"gt:{int(label)}", fill="lime")

    for pred in pred_boxes:
        x1, y1, x2, y2 = pred["box"]
        score = pred["score"]
        class_id = pred["class_id"]
        gy, gx = pred["grid"]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1 + 2, max(0.0, y1 - 12)), f"pred:{class_id} {score:.2f} ({gy},{gx})", fill="red")

    return image


def save_progress_visualizations(
    model,
    dataset,
    device,
    output_root,
    epoch,
    image_size,
    grid_size,
    num_classes,
    top_k,
    score_threshold,
    max_samples,
):
    """Save fixed validation-sample prediction snapshots for training-progress visualization."""
    model.eval()
    output_root = Path(output_root)
    epoch_dir = output_root / f"epoch_{epoch:03d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for index in range(min(max_samples, len(dataset))):
            image, target = dataset[index]
            pred = model(image.unsqueeze(0).to(device))[0].cpu()
            pred_boxes = decode_predictions(
                pred=pred,
                image_size=image_size,
                grid_size=grid_size,
                num_classes=num_classes,
                top_k=top_k,
                score_threshold=score_threshold,
            )

            vis_image = tensor_to_pil(image)
            vis_image = draw_boxes(
                image=vis_image,
                gt_boxes=target["boxes"],
                gt_labels=target["labels"],
                pred_boxes=pred_boxes,
            )

            sample_id = target["sample_id"]
            save_path = epoch_dir / f"{sample_id}.png"
            vis_image.save(save_path)

    return epoch_dir
