import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BLOCKS = ("V1", "V2", "V4", "IT")


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    rows = read_rows(args.results / "adaptive_summary.csv")
    names = [row["name"] for row in rows]
    x = np.arange(len(names))

    figure, axes = plt.subplots(2, 2, figsize=(14, 11), constrained_layout=True)
    width = 0.18
    for index, block in enumerate(BLOCKS):
        key = f"sigma_{block.lower()}"
        means = [float(row[f"{key}_mean"]) for row in rows]
        errors = [float(row[f"{key}_std"]) for row in rows]
        axes[0, 0].bar(
            x + (index - 1.5) * width,
            means,
            width,
            yerr=errors,
            capsize=3,
            label=block,
        )
    axes[0, 0].set_title("Итоговые профили шума")
    axes[0, 0].set_xticks(x, names)
    axes[0, 0].set_ylabel("Интенсивность")
    axes[0, 0].legend()
    axes[0, 0].grid(axis="y", alpha=0.25)

    metrics = (
        ("clean_accuracy", "Clean accuracy"),
        ("transfer_pgd_eps2", "Transfer PGD, ε=2/255"),
        ("transfer_pgd_eps4", "Transfer PGD, ε=4/255"),
        ("transfer_pgd_eps8", "Transfer PGD, ε=8/255"),
    )
    for axis, (metric, title) in zip(
        (axes[0, 1], axes[1, 0], axes[1, 1]), metrics[1:]
    ):
        means = [100 * float(row[f"{metric}_mean"]) for row in rows]
        errors = [100 * float(row[f"{metric}_std"]) for row in rows]
        axis.bar(names, means, yerr=errors, capsize=4)
        axis.set_title(title)
        axis.set_ylabel("Accuracy, %")
        axis.grid(axis="y", alpha=0.25)

    output = args.results / "adaptive_comparison.png"
    figure.savefig(output, dpi=180)
    plt.close(figure)
    print(f"Создан {output}")


if __name__ == "__main__":
    main()
