import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="Export TensorBoard loss curves for MiniYOLO.")
    parser.add_argument(
        "--run-dir",
        type=str,
        default=str(PROJECT_ROOT / "runs" / "real_train"),
        help="TensorBoard run directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "training_curves"),
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


def plot_group(curves, tags, title, ylabel, output_path):
    plt.figure(figsize=(9, 5))
    for tag in tags:
        if tag not in curves:
            continue
        steps = curves[tag]["steps"]
        values = curves[tag]["values"]
        plt.plot(steps, values, marker="o", label=tag)

    plt.title(title)
    plt.xlabel("Step / Epoch")
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
        title="MiniYOLO Step Loss Curves",
        ylabel="Loss",
        output_path=output_dir / "loss_step_curves.png",
    )

    plot_group(
        curves,
        tags=["loss/total_epoch", "loss/classification_epoch", "loss/box_epoch"],
        title="MiniYOLO Epoch Loss Curves",
        ylabel="Loss",
        output_path=output_dir / "loss_epoch_curves.png",
    )

    summary = {
        "event_file": str(event_file),
        "available_tags": sorted(curves.keys()),
        "final_values": {
            tag: curves[tag]["values"][-1] for tag in curves if curves[tag]["values"]
        },
    }
    (output_dir / "training_curve_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Exported training curves to {output_dir}")


if __name__ == "__main__":
    main()
