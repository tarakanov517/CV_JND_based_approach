import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


LAYERS = ("V1", "V2", "V4", "IT", "all")
LABELS = ("V1", "V2", "V4", "IT", "Все блоки")


def read_summary(path):
    with path.open(newline="", encoding="utf-8") as file:
        return {row["name"]: row for row in csv.DictReader(file)}


def values(summary, kind, metric):
    rows = [summary[f"{kind}_{layer.lower()}"] for layer in LAYERS]
    means = np.array([100 * float(row[f"{metric}_mean"]) for row in rows])
    stds = np.array([100 * float(row[f"{metric}_std"]) for row in rows])
    return means, stds


def draw_metric(axis, summary, kind, title, metric):
    for epsilon, color in ((2, "#2563eb"), (4, "#dc2626")):
        mean, std = values(summary, kind, f"{metric}_eps{epsilon}")
        baseline = 100 * float(summary["baseline"][f"{metric}_eps{epsilon}_mean"])
        axis.errorbar(
            LABELS,
            mean,
            yerr=std,
            marker="o",
            capsize=3,
            color=color,
            label=f"ε={epsilon}/255",
        )
        axis.axhline(
            baseline,
            linestyle="--",
            color=color,
            alpha=0.55,
            label=f"Baseline ε={epsilon}/255",
        )
    axis.set_title(title)
    axis.set_ylabel("Accuracy, %")
    axis.grid(alpha=0.25)
    axis.legend()


def plot_kind(summary, kind, output_dir):
    figure, axes = plt.subplots(3, 2, figsize=(13, 14), constrained_layout=True)
    clean, clean_std = values(summary, kind, "clean_accuracy")
    axes[0, 0].errorbar(
        LABELS, clean, yerr=clean_std, marker="o", capsize=3, color="#111827"
    )
    axes[0, 0].set_title("Чистые изображения")
    axes[0, 0].set_ylabel("Accuracy, %")
    axes[0, 0].grid(alpha=0.25)

    baseline = summary["baseline"]
    baseline_clean = 100 * float(baseline["clean_accuracy_mean"])
    axes[0, 0].axhline(
        baseline_clean, linestyle="--", color="#6b7280", label="Baseline"
    )
    axes[0, 0].legend()

    draw_metric(axes[0, 1], summary, kind, "White-box FGSM", "white_fgsm")
    draw_metric(axes[1, 0], summary, kind, "White-box PGD", "white_pgd")
    draw_metric(axes[1, 1], summary, kind, "Transfer FGSM", "transfer_fgsm")
    draw_metric(axes[2, 0], summary, kind, "Transfer PGD", "transfer_pgd")
    axes[2, 1].axis("off")

    title = "Аксональный шум" if kind == "axon" else "Дендритный шум"
    figure.suptitle(f"{title}: влияние расположения", fontsize=16)
    output = output_dir / f"layerwise_{kind}.png"
    figure.savefig(output, dpi=180)
    plt.close(figure)
    print(f"Создан {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    summary = read_summary(args.results / "layerwise_summary.csv")
    plot_kind(summary, "axon", args.results)
    plot_kind(summary, "dendrite", args.results)


if __name__ == "__main__":
    main()
