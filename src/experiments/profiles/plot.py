import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BLOCKS = ("V1", "V2", "V4", "IT")
PROFILE_NAMES = (
    "uniform_040",
    "rising_budget",
    "falling_budget",
    "early_loaded",
    "late_loaded",
    "middle_peak",
    "edge_peak",
    "uniform_050",
    "rising_high",
    "falling_high",
    "uniform_055",
)


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def value(row, metric):
    return 100 * float(row[f"{metric}_mean"])


def plot_profiles(rows, output_dir):
    selected = [row for name in PROFILE_NAMES for row in rows if row["name"] == name]
    figure, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)

    for row in selected:
        sigmas = [float(row[f"sigma_{block.lower()}"]) for block in BLOCKS]
        axes[0].plot(BLOCKS, sigmas, marker="o", label=row["name"])
    axes[0].set_title("Профили аксонального шума")
    axes[0].set_ylabel("Интенсивность")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8, ncol=2)

    clean = [value(row, "clean_accuracy") for row in selected]
    pgd4 = [value(row, "transfer_pgd_eps4") for row in selected]
    axes[1].scatter(clean, pgd4, s=55)
    for row, x, y in zip(selected, clean, pgd4):
        axes[1].annotate(row["name"], (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8)
    axes[1].set_title("Компромисс качества и transfer PGD, ε=4/255")
    axes[1].set_xlabel("Clean accuracy, %")
    axes[1].set_ylabel("Transfer PGD accuracy, %")
    axes[1].grid(alpha=0.25)

    output = output_dir / "profile_comparison.png"
    figure.savefig(output, dpi=180)
    plt.close(figure)
    print(f"Создан {output}")


def grid_row(rows, v1, v2):
    for row in rows:
        if (
            float(row["sigma_v1"]) == v1
            and float(row["sigma_v2"]) == v2
            and float(row["sigma_v4"]) == 0.0
            and float(row["sigma_it"]) == 0.0
        ):
            return row
    raise ValueError(f"Не найдена конфигурация V1={v1}, V2={v2}")


def plot_interaction(rows, output_dir):
    levels = (0.0, 0.4, 0.75)
    metrics = (
        ("clean_accuracy", "Clean accuracy"),
        ("transfer_pgd_eps2", "Transfer PGD, ε=2/255"),
        ("transfer_pgd_eps4", "Transfer PGD, ε=4/255"),
        ("transfer_pgd_eps8", "Transfer PGD, ε=8/255"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    for axis, (metric, title) in zip(axes.flat, metrics):
        matrix = np.array(
            [[value(grid_row(rows, v1, v2), metric) for v2 in levels] for v1 in levels]
        )
        image = axis.imshow(matrix, cmap="viridis", origin="lower")
        for row_index in range(len(levels)):
            for column_index in range(len(levels)):
                axis.text(
                    column_index,
                    row_index,
                    f"{matrix[row_index, column_index]:.1f}",
                    ha="center",
                    va="center",
                    color="white" if matrix[row_index, column_index] < matrix.max() * 0.7 else "black",
                )
        axis.set_xticks(range(len(levels)), levels)
        axis.set_yticks(range(len(levels)), levels)
        axis.set_xlabel("Интенсивность V2")
        axis.set_ylabel("Интенсивность V1")
        axis.set_title(title)
        figure.colorbar(image, ax=axis, label="Accuracy, %")

    output = output_dir / "v1_v2_interaction.png"
    figure.savefig(output, dpi=180)
    plt.close(figure)
    print(f"Создан {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    rows = read_rows(args.results / "profiles_summary.csv")
    plot_profiles(rows, args.results)
    plot_interaction(rows, args.results)


if __name__ == "__main__":
    main()
