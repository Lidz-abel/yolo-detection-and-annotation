"""Plot train/validation loss curves from one recorded result.txt."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Plot loss curves from a YOLO result.txt file.")
    parser.add_argument("--result-txt", type=str, required=True, help="Path to logs/records/<run_id>/result.txt.")
    parser.add_argument("--output-png", type=str, required=True, help="Output PNG path.")
    parser.add_argument("--title", type=str, default="Loss Curves", help="Figure title.")
    return parser.parse_args()


def parse_history(section: str):
    rows = []
    pattern = re.compile(
        r"epoch =\s*(\d+) \| total = ([0-9.]+) \| box = ([0-9.]+) \| obj = ([0-9.]+) \| cls = ([0-9.]+)"
    )
    for line in section.splitlines():
        match = pattern.search(line)
        if match:
            rows.append(tuple([int(match.group(1))] + [float(match.group(index)) for index in range(2, 6)]))
    return rows


def main():
    args = parse_args()
    result_path = Path(args.result_txt)
    text = result_path.read_text(encoding="utf-8")
    train_section = text.split("train history:", 1)[1].split("val history:", 1)[0]
    val_section = text.split("val history:", 1)[1].split("artifacts:", 1)[0]
    train = parse_history(train_section)
    val = parse_history(val_section)
    if not train or not val:
        raise ValueError(f"No train/val history found in {result_path}")

    output_path = Path(args.output_png)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    names = ["total", "box", "obj", "cls"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=160)
    axes = axes.ravel()
    train_epochs = [row[0] for row in train]
    val_epochs = [row[0] for row in val]
    for index, name in enumerate(names):
        ax = axes[index]
        ax.plot(train_epochs, [row[index + 1] for row in train], label="train", linewidth=2)
        ax.plot(val_epochs, [row[index + 1] for row in val], label="val", linewidth=2)
        ax.set_title(f"{name} loss")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle(args.title, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path)
    print(output_path)


if __name__ == "__main__":
    main()
