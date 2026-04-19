from collections import defaultdict

import torch

from utils.prediction import box_iou_xyxy, decode_predictions


def _compute_ap(recalls, precisions):
    if len(recalls) == 0:
        return 0.0

    mrec = [0.0] + recalls + [1.0]
    mpre = [0.0] + precisions + [0.0]

    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    ap = 0.0
    for i in range(len(mrec) - 1):
        ap += (mrec[i + 1] - mrec[i]) * mpre[i + 1]
    return ap


def evaluate_map50(model, loader, device, image_size, grid_size, num_classes, iou_threshold, top_k, score_threshold):
    """Evaluate a lightweight mAP@0.5 on the validation loader."""
    model.eval()

    gt_by_class = defaultdict(lambda: defaultdict(list))
    pred_by_class = defaultdict(list)
    gt_count_by_class = defaultdict(int)
    pred_score_sum = 0.0
    pred_count = 0

    with torch.no_grad():
        sample_index = 0
        for images, targets in loader:
            images = images.to(device)
            preds = model(images).cpu()

            batch_size = images.shape[0]
            for batch_idx in range(batch_size):
                sample_id = sample_index
                pred_boxes = decode_predictions(
                    pred=preds[batch_idx],
                    image_size=image_size,
                    grid_size=grid_size,
                    num_classes=num_classes,
                    top_k=top_k,
                    score_threshold=score_threshold,
                )

                gt_boxes = targets["boxes"][batch_idx].float()
                gt_labels = targets["labels"][batch_idx].long()

                for gt_box, gt_label in zip(gt_boxes, gt_labels):
                    class_id = int(gt_label.item())
                    gt_by_class[class_id][sample_id].append(gt_box)
                    gt_count_by_class[class_id] += 1

                for pred_item in pred_boxes:
                    pred_by_class[pred_item["class_id"]].append(
                        {
                            "sample_id": sample_id,
                            "score": pred_item["score"],
                            "box": torch.tensor(pred_item["box"], dtype=torch.float32),
                        }
                    )
                    pred_score_sum += pred_item["score"]
                    pred_count += 1

                sample_index += 1

    ap_by_class = {}
    present_classes = sorted(gt_count_by_class.keys())

    for class_id in present_classes:
        preds = pred_by_class[class_id]
        preds.sort(key=lambda item: item["score"], reverse=True)

        matched = {
            sample_id: [False] * len(boxes)
            for sample_id, boxes in gt_by_class[class_id].items()
        }

        tp = []
        fp = []

        for pred_item in preds:
            sample_id = pred_item["sample_id"]
            pred_box = pred_item["box"].unsqueeze(0)
            gt_boxes = gt_by_class[class_id].get(sample_id, [])

            if not gt_boxes:
                tp.append(0.0)
                fp.append(1.0)
                continue

            gt_tensor = torch.stack(gt_boxes, dim=0)
            ious = box_iou_xyxy(pred_box, gt_tensor)[0]
            best_iou, best_idx = ious.max(dim=0)
            best_idx = int(best_idx.item())

            if best_iou.item() >= iou_threshold and not matched[sample_id][best_idx]:
                matched[sample_id][best_idx] = True
                tp.append(1.0)
                fp.append(0.0)
            else:
                tp.append(0.0)
                fp.append(1.0)

        if len(tp) == 0:
            ap_by_class[class_id] = 0.0
            continue

        tp_cum = []
        fp_cum = []
        running_tp = 0.0
        running_fp = 0.0
        for tp_value, fp_value in zip(tp, fp):
            running_tp += tp_value
            running_fp += fp_value
            tp_cum.append(running_tp)
            fp_cum.append(running_fp)

        recalls = [value / max(gt_count_by_class[class_id], 1) for value in tp_cum]
        precisions = [tp_v / max(tp_v + fp_v, 1e-6) for tp_v, fp_v in zip(tp_cum, fp_cum)]
        ap_by_class[class_id] = _compute_ap(recalls, precisions)

    if present_classes:
        map50 = sum(ap_by_class[class_id] for class_id in present_classes) / len(present_classes)
    else:
        map50 = 0.0

    mean_pred_score = pred_score_sum / pred_count if pred_count > 0 else 0.0
    mean_pred_boxes = pred_count / max(sum(len(sample_boxes) for sample_boxes in gt_by_class[present_classes[0]].values()), 1) if False else pred_count

    return {
        "map50": map50,
        "ap_by_class": ap_by_class,
        "num_present_classes": len(present_classes),
        "num_predictions": pred_count,
        "mean_pred_score": mean_pred_score,
    }
