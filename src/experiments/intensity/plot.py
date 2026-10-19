import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


SINGLE_BLOCKS = ("V1", "V2", "V4", "IT")
COMBINATIONS = ("V1+V2", "V1+V4", "V1+IT", "V2+V4", "V2+IT", "V4+IT", "all")


def read_summary(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def baseline_value(rows, metric):
    row = next(row for row in rows if row["noise_kind"] == "baseline")
    return 100 * float(row[f"{metric}_mean"])


def block_rows(rows, kind, block):
    selected = [
        row
        for row in rows
        if row["noise_kind"] == kind and row["noise_block"] == block
    ]
    return sorted(selected, key=lambda row: float(row["intensity"]))


def draw(axis, rows, kind, blocks, metric, title):
    baseline = baseline_value(rows, metric)
    axis.axhline(
        baseline,
        color="#6b7280",
        linestyle="--",
        linewidth=1.5,
        label="Baseline",
    )
    for block in blocks:
        selected = block_rows(rows, kind, block)
        levels = [float(row["intensity"]) for row in selected]
        means = [100 * float(row[f"{metric}_mean"]) for row in selected]
        axis.plot(
            levels,
            means,
            marker="o",
            linewidth=2,
            label="Все блоки" if block == "all" else block,
        )
    axis.set_title(title)
    axis.set_xlabel("Интенсивность шума")
    axis.set_ylabel("Accuracy, %")
    axis.grid(alpha=0.25)
    axis.legend()


def plot_group(rows, kind, blocks, title, filename, output_dir):
    figure, axes = plt.subplots(4, 2, figsize=(14, 20), constrained_layout=True)
    draw(axes[0, 0], rows, kind, blocks, "clean_accuracy", "Чистые изображения")
    axes[0, 1].axis("off")
    draw(
        axes[1, 0],
        rows,
        kind,
        blocks,
        "transfer_fgsm_eps2",
        "Transfer FGSM, ε=2/255",
    )
    draw(
        axes[1, 1],
        rows,
        kind,
        blocks,
        "transfer_pgd_eps2",
        "Transfer PGD, ε=2/255",
    )
    draw(
        axes[2, 0],
        rows,
        kind,
        blocks,
        "transfer_fgsm_eps4",
        "Transfer FGSM, ε=4/255",
    )
    draw(
        axes[2, 1],
        rows,
        kind,
        blocks,
        "transfer_pgd_eps4",
        "Transfer PGD, ε=4/255",
    )
    draw(
        axes[3, 0],
        rows,
        kind,
        blocks,
        "transfer_fgsm_eps8",
        "Transfer FGSM, ε=8/255",
    )
    draw(
        axes[3, 1],
        rows,
        kind,
        blocks,
        "transfer_pgd_eps8",
        "Transfer PGD, ε=8/255",
    )
    figure.suptitle(title, fontsize=17)
    output = output_dir / filename
    figure.savefig(output, dpi=180)
    plt.close(figure)
    print(f"Создан {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    rows = read_summary(args.results / "intensity_summary.csv")
    plot_group(
        rows,
        "axon",
        SINGLE_BLOCKS,
        "Аксональный шум: отдельные блоки",
        "intensity_axon_single.png",
        args.results,
    )
    plot_group(
        rows,
        "axon",
        COMBINATIONS,
        "Аксональный шум: пары и все блоки",
        "intensity_axon_combinations.png",
        args.results,
    )


if __name__ == "__main__":
    main()
