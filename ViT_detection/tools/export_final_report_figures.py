from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "report_assets" / "final_version1"
OUT.mkdir(parents=True, exist_ok=True)

font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
font_manager.fontManager.addfont(font_path)
plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False


runs = [
    "COCO-only 320",
    "320 topk1",
    "416 原anchor",
    "416 K6",
    "K6 tuned",
    "P3/P4/P5 K9",
    "P3 K9 area",
]
ap = [0.126985, 0.108180, 0.148786, 0.146778, 0.147754, 0.136481, 0.120353]
ap50 = [0.218369, 0.195980, 0.248717, 0.244778, 0.247426, 0.228555, 0.198501]
ar100 = [0.260567, 0.239071, 0.302299, 0.303445, 0.323725, 0.313965, 0.298278]
best_val = [5.262634, 5.148357, 4.779924, 4.694147, None, 6.605443, 4.485392]


def save_metric_bars():
    x = range(len(runs))
    width = 0.26
    fig, ax = plt.subplots(figsize=(12, 5.6))
    ax.bar([i - width for i in x], ap, width=width, label="AP")
    ax.bar(list(x), ap50, width=width, label="AP50")
    ax.bar([i + width for i in x], ar100, width=width, label="AR100")
    ax.set_ylim(0.0, 0.36)
    ax.set_ylabel("COCO metric")
    ax.set_title("已完成正式实验 COCO 指标对比")
    ax.set_xticks(list(x))
    ax.set_xticklabels(runs, rotation=24, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(ncol=3, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "completed_coco_metrics.png", dpi=180)
    plt.close(fig)


def save_val_loss_bars():
    labels = [r for r, v in zip(runs, best_val) if v is not None]
    vals = [v for v in best_val if v is not None]
    colors = ["#5577aa", "#5577aa", "#2f9e44", "#2f9e44", "#aa7755", "#aa7755"]
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ax.bar(labels, vals, color=colors)
    ax.set_ylabel("Best val loss")
    ax.set_title("已完成训练实验 best validation loss")
    ax.set_ylim(4.2, 6.9)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=24, ha="right")
    for i, value in enumerate(vals):
        ax.text(i, value + 0.04, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "completed_val_loss.png", dpi=180)
    plt.close(fig)


def save_route_plot():
    route_labels = [
        "评估修复",
        "COCO-only",
        "DDP topk1",
        "416",
        "K6",
        "K6 tuned",
        "P3 K9",
        "P3 area",
    ]
    route_ap = [0.121699, 0.126985, 0.108180, 0.148786, 0.146778, 0.147754, 0.136481, 0.120353]
    route_ar = [0.263924, 0.260567, 0.239071, 0.302299, 0.303445, 0.323725, 0.313965, 0.298278]
    x = range(len(route_labels))
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.plot(x, route_ap, marker="o", linewidth=2.2, label="AP")
    ax.plot(x, route_ar, marker="s", linewidth=2.2, label="AR100")
    ax.set_ylim(0.08, 0.35)
    ax.set_ylabel("COCO metric")
    ax.set_title("实验路线中的 AP 与 AR100 变化")
    ax.set_xticks(list(x))
    ax.set_xticklabels(route_labels, rotation=24, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "route_ap_ar100.png", dpi=180)
    plt.close(fig)


def main():
    save_metric_bars()
    save_val_loss_bars()
    save_route_plot()
    summary = OUT / "figure_export_summary.txt"
    summary.write_text(
        "\n".join(
            [
                "completed_coco_metrics.png",
                "completed_val_loss.png",
                "route_ap_ar100.png",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
