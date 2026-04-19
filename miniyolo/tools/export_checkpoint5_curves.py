import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="Export TensorBoard curves for checkpoint 5 training.")
    parser.add_argument(
        "--run-dir",
        type=str,
        default=str(PROJECT_ROOT / "runs" / "checkpoint5_fulltrain"),
        help="TensorBoard run directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "checkpoint5" / "curves"),
        help="Directory to save exported plots.",
    )
    return parser.parse_args()


def load_scalars(run_dir):
    run_path = Path(run_dir)
    event_files = sorted(run_path.glob("events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event file found in {run_dir}")

    event_file = event_files[-1]
    accumulator = EventAccumulator(str(event_file))
    accumulator.Reload()

    scalars = {}
    for tag in accumulator.Tags().get("scalars", []):
        values = accumulator.Scalars(tag)
        scalars[tag] = {
            "steps": [item.step for item in values],
            "values": [item.value for item in values],
        }
    return event_file, scalars


def plot_group(curves, tags, title, ylabel, output_path, xlabel="Step"):
    plt.figure(figsize=(9, 5))
    for tag in tags:
        if tag not in curves:
            continue
        plt.plot(curves[tag]["steps"], curves[tag]["values"], marker="o", label=tag)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    event_file, curves = load_scalars(args.run_dir)

    plot_group(
        curves,
        tags=["loss/total_step", "loss/classification_step", "loss/box_step"],
        title="Checkpoint 5 Step Loss Curves",
        ylabel="Loss",
        output_path=output_dir / "loss_step_curves.png",
        xlabel="Step",
    )

    plot_group(
        curves,
        tags=["loss/total_epoch", "loss/classification_epoch", "loss/box_epoch"],
        title="Checkpoint 5 Epoch Loss Curves",
        ylabel="Loss",
        output_path=output_dir / "loss_epoch_curves.png",
        xlabel="Epoch",
    )

    plot_group(
        curves,
        tags=["train/lr"],
        title="Checkpoint 5 Step Learning Rate Curve",
        ylabel="Learning Rate",
        output_path=output_dir / "learning_rate_step_curve.png",
        xlabel="Step",
    )

    plot_group(
        curves,
        tags=["train/lr_epoch"],
        title="Checkpoint 5 Epoch Learning Rate Curve",
        ylabel="Learning Rate",
        output_path=output_dir / "learning_rate_epoch_curve.png",
        xlabel="Epoch",
    )

    plot_group(
        curves,
        tags=["val/map50"],
        title="Checkpoint 5 Validation mAP@0.5",
        ylabel="mAP@0.5",
        output_path=output_dir / "map50_curve.png",
        xlabel="Epoch",
    )

    plot_group(
        curves,
        tags=["grad_norm/global_total"],
        title="Checkpoint 5 Global Gradient Norm",
        ylabel="Gradient Norm",
        output_path=output_dir / "grad_global_curve.png",
        xlabel="Step",
    )

    plot_group(
        curves,
        tags=["val/mean_pred_score"],
        title="Checkpoint 5 Mean Prediction Score",
        ylabel="Mean Score",
        output_path=output_dir / "mean_pred_score_curve.png",
        xlabel="Epoch",
    )

    plot_group(
        curves,
        tags=["val/num_predictions"],
        title="Checkpoint 5 Number of Predictions",
        ylabel="Number of Predictions",
        output_path=output_dir / "num_predictions_curve.png",
        xlabel="Epoch",
    )

    plot_group(
        curves,
        tags=["train/positive_cells_per_image"],
        title="Checkpoint 5 Positive Cells per Image",
        ylabel="Positive Cells per Image",
        output_path=output_dir / "positive_cells_curve.png",
        xlabel="Step",
    )

    summary = {
        "event_file": str(event_file),
        "available_tags": sorted(curves.keys()),
        "final_values": {
            tag: curves[tag]["values"][-1]
            for tag in curves
            if curves[tag]["values"]
        },
    }
    (output_dir / "checkpoint5_curve_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Exported checkpoint 5 curves to {output_dir}")


if __name__ == "__main__":
    main()
