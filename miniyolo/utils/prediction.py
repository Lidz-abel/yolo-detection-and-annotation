import torch


def decode_predictions(pred, image_size, grid_size, num_classes, top_k, score_threshold):
    """Decode one prediction map [7, 7, num_classes + 4] into ranked boxes."""
    class_logits = pred[..., :num_classes]
    box_pred = pred[..., num_classes:]

    class_probs = torch.softmax(class_logits, dim=-1)
    scores, class_ids = class_probs.max(dim=-1)
    box_pred = torch.sigmoid(box_pred)

    candidates = []
    for gy in range(grid_size):
        for gx in range(grid_size):
            score = scores[gy, gx].item()
            if score < score_threshold:
                continue

            cell_x, cell_y, w, h = box_pred[gy, gx].tolist()
            cx = (gx + cell_x) / grid_size * image_size
            cy = (gy + cell_y) / grid_size * image_size
            bw = w * image_size
            bh = h * image_size

            x1 = max(0.0, cx - bw / 2.0)
            y1 = max(0.0, cy - bh / 2.0)
            x2 = min(float(image_size), cx + bw / 2.0)
            y2 = min(float(image_size), cy + bh / 2.0)

            candidates.append(
                {
                    "box": [x1, y1, x2, y2],
                    "score": score,
                    "class_id": int(class_ids[gy, gx].item()),
                    "grid": (gy, gx),
                }
            )

    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates[:top_k]


def box_iou_xyxy(boxes1, boxes2):
    """Compute IoU matrix between two xyxy box tensors."""
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=torch.float32)

    tl = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    br = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (br - tl).clamp(min=0.0)
    inter = wh[..., 0] * wh[..., 1]

    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0.0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0.0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0.0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0.0)
    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp(min=1e-6)
