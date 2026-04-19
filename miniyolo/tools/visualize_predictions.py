import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from data.detection_dataset import DetectionDataset
from models.miniyolo import MiniYOLO
from utils.prediction import decode_predictions
from utils.visualization import draw_boxes, tensor_to_pil


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize MiniYOLO predictions against ground-truth boxes.")
    parser.add_argument(
        "--manifest",
        type=str,
        default="/home/lidz/YOLO/DataSet/Unified/manifests/all_val.jsonl",
        help="Path to unified manifest jsonl.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "miniyolo_real_last.pth"),
        help="Path to trained model weights.",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--grid-size", type=int, default=7)
    parser.add_argument("--num-classes", type=int, default=20)
    parser.add_argument("--max-samples", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--score-threshold", type=float, default=0.20)
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = DetectionDataset(
        manifest_path=args.manifest,
        image_size=args.image_size,
        grid_size=args.grid_size,
        num_classes=args.num_classes,
        max_samples=args.max_samples,
    )

    model = MiniYOLO(num_classes=args.num_classes).to(device)
    state_dict = torch.load(args.weights, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    output_dir = PROJECT_ROOT / "outputs" / "prediction_visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("device:", device)
    print("weights:", args.weights)
    print("output_dir:", output_dir)

    with torch.no_grad():
        for index in range(len(dataset)):
            image, target = dataset[index]
            pred = model(image.unsqueeze(0).to(device))[0].cpu()

            pred_boxes = decode_predictions(
                pred=pred,
                image_size=args.image_size,
                grid_size=args.grid_size,
                num_classes=args.num_classes,
                top_k=args.top_k,
                score_threshold=args.score_threshold,
            )

            vis_image = tensor_to_pil(image)
            vis_image = draw_boxes(
                image=vis_image,
                gt_boxes=target["boxes"],
                gt_labels=target["labels"],
                pred_boxes=pred_boxes,
            )

            sample_id = target["sample_id"]
            save_path = output_dir / f"{sample_id}.png"
            vis_image.save(save_path)

            print(f"[{index}] sample_id={sample_id} | gt_boxes={len(target['boxes'])} | pred_boxes={len(pred_boxes)}")
            print("saved to:", save_path)


if __name__ == "__main__":
    main()
