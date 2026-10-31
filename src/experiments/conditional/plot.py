import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    rows = read_rows(args.results / "conditional_summary.csv")
    names = [row["profile"] for row in rows]
    metrics = (
        ("clean_accuracy", "Clean accuracy"),
        ("transfer_pgd_eps2", "Transfer PGD, ε=2/255"),
        ("transfer_pgd_eps4", "Transfer PGD, ε=4/255"),
        ("transfer_pgd_eps8", "Transfer PGD, ε=8/255"),
        ("white_pgd_eps4", "EOT white-box PGD, ε=4/255"),
        ("white_pgd_eps8", "EOT white-box PGD, ε=8/255"),
    )
    figure, axes = plt.subplots(2, 3, figsize=(20, 11), constrained_layout=True)
    for axis, (metric, title) in zip(axes.flat, metrics):
        means = [100 * float(row[f"{metric}_mean"]) for row in rows]
        errors = [100 * float(row[f"{metric}_std"]) for row in rows]
        axis.bar(names, means, yerr=errors, capsize=3)
        axis.set_title(title)
        axis.set_ylabel("Accuracy, %")
        axis.tick_params(axis="x", rotation=35)
        axis.grid(axis="y", alpha=0.25)
    output = args.results / "conditional_evaluation.png"
    figure.savefig(output, dpi=180)
    plt.close(figure)
    print(f"Создан {output}")

    history_paths = sorted(
        args.results.glob("conditional_axon/seed_*/history.csv")
    )
    if not history_paths:
        return
    figure, axes = plt.subplots(
        1, len(history_paths), figsize=(18, 5), squeeze=False, constrained_layout=True
    )
    for axis, path in zip(axes.flat, history_paths):
        history = read_rows(path)
        epochs = [int(row["epoch"]) for row in history]
        zero = [100 * float(row["validation_zero_accuracy"]) for row in history]
        anchor = [100 * float(row["validation_anchor_accuracy"]) for row in history]
        axis.plot(epochs, zero, label="σ=0")
        axis.plot(epochs, anchor, label="σ=0.85 во всех блоках")
        axis.set_title(path.parent.name)
        axis.set_xlabel("Эпоха")
        axis.set_ylabel("Validation accuracy, %")
        axis.grid(alpha=0.25)
        axis.legend()
    output = args.results / "conditional_training.png"
    figure.savefig(output, dpi=180)
    plt.close(figure)
    print(f"Создан {output}")


if __name__ == "__main__":
    main()
