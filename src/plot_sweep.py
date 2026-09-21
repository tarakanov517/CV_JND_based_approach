import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


NOISES = ["lateral", "contrast", "pyramidal", "axon", "dendrite"]
TITLES = {
    "lateral": "Латеральный шум",
    "contrast": "Контрастно-адаптивный шум",
    "pyramidal": "Пирамидальный шум",
    "axon": "Аксональный шум",
    "dendrite": "Дендритный шум",
}


def read_training(results_dir):
    rows = []
    for path in results_dir.glob("*/seed_*/result.json"):
        with path.open(encoding="utf-8") as file:
            rows.append(json.load(file))
    return rows


def read_transfer(results_dir):
    values = {}
    for path in results_dir.glob("transfer_seed_*.csv"):
        with path.open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                values[(row["name"], int(row["seed"]))] = row
    return values


def mean_std(values):
    array = np.asarray(values, dtype=float)
    std = array.std(ddof=1) if len(array) > 1 else 0.0
    return float(array.mean()), float(std)


def collect(rows, transfer):
    groups = defaultdict(list)
    for row in rows:
        key = (row["sweep_kind"], float(row["sweep_level"]))
        transfer_row = transfer.get((row["name"], int(row["seed"])), {})
        groups[key].append(
            {
                "clean": row["clean_accuracy"],
                "white_fgsm": row["fgsm_accuracy"],
                "white_pgd": row["pgd_accuracy"],
                "transfer_fgsm": float(
                    transfer_row.get("baseline_fgsm_accuracy", "nan")
                ),
                "transfer_pgd": float(
                    transfer_row.get("baseline_pgd_accuracy", "nan")
                ),
            }
        )
    summary = []
    for (kind, level), group in sorted(groups.items()):
        item = {"sweep_kind": kind, "sweep_level": level, "seeds": len(group)}
        for metric in group[0]:
            values = [row[metric] for row in group if not np.isnan(row[metric])]
            item[f"{metric}_mean"], item[f"{metric}_std"] = mean_std(values)
        summary.append(item)
    return summary


def detailed_rows(rows, transfer):
    output = []
    for row in sorted(
        rows,
        key=lambda item: (
            item["sweep_kind"],
            float(item["sweep_level"]),
            int(item["seed"]),
        ),
    ):
        transfer_row = transfer.get((row["name"], int(row["seed"])), {})
        output.append(
            {
                "name": row["name"],
                "sweep_kind": row["sweep_kind"],
                "sweep_level": row["sweep_level"],
                "seed": row["seed"],
                "best_validation_accuracy": row["best_validation_accuracy"],
                "clean_accuracy": row["clean_accuracy"],
                "white_fgsm_accuracy": row["fgsm_accuracy"],
                "white_pgd_accuracy": row["pgd_accuracy"],
                "transfer_fgsm_accuracy": transfer_row.get(
                    "baseline_fgsm_accuracy", ""
                ),
                "transfer_pgd_accuracy": transfer_row.get(
                    "baseline_pgd_accuracy", ""
                ),
            }
        )
    return output


def write_summary(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def series(rows, kind, metric):
    selected = sorted(
        (row for row in rows if row["sweep_kind"] == kind),
        key=lambda row: row["sweep_level"],
    )
    x = np.array([row["sweep_level"] for row in selected])
    mean = 100 * np.array([row[f"{metric}_mean"] for row in selected])
    std = 100 * np.array([row[f"{metric}_std"] for row in selected])
    return x, mean, std


def plot(results_dir, rows):
    baseline = next(row for row in rows if row["sweep_kind"] == "baseline")
    figure, axes = plt.subplots(len(NOISES), 2, figsize=(13, 18), constrained_layout=True)
    for index, kind in enumerate(NOISES):
        clean_ax, attack_ax = axes[index]
        x, clean, clean_std = series(rows, kind, "clean")
        clean_ax.errorbar(x, clean, yerr=clean_std, marker="o", capsize=3)
        clean_ax.axhline(
            100 * baseline["clean_mean"], color="black", linestyle="--", label="Baseline"
        )
        clean_ax.set_title(f"{TITLES[kind]}: clean")
        clean_ax.set_ylabel("Accuracy, %")
        clean_ax.grid(alpha=0.25)
        clean_ax.legend()

        for metric, label, marker in [
            ("white_fgsm", "White-box FGSM", "o"),
            ("transfer_fgsm", "Baseline-transfer FGSM", "s"),
            ("white_pgd", "White-box PGD", "^"),
            ("transfer_pgd", "Baseline-transfer PGD", "D"),
        ]:
            x, mean, std = series(rows, kind, metric)
            attack_ax.errorbar(x, mean, yerr=std, marker=marker, capsize=3, label=label)
        attack_ax.set_title(f"{TITLES[kind]}: атаки")
        attack_ax.set_ylabel("Accuracy, %")
        attack_ax.grid(alpha=0.25)
        attack_ax.legend(fontsize=8)

        x_label = "Множитель k" if kind == "contrast" else "Интенсивность шума"
        clean_ax.set_xlabel(x_label)
        attack_ax.set_xlabel(x_label)
    figure.suptitle("Зависимость accuracy от интенсивности человекоподобного шума")
    output = results_dir / "noise_accuracy_curves.png"
    figure.savefig(output, dpi=180)
    print(f"Создан {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    args = parser.parse_args()
    rows = read_training(args.results_dir)
    transfer = read_transfer(args.results_dir)
    if not rows:
        raise SystemExit("Не найдены result.json")
    details = detailed_rows(rows, transfer)
    summary = collect(rows, transfer)
    write_summary(args.results_dir / "sweep_results.csv", details)
    write_summary(args.results_dir / "sweep_summary.csv", summary)
    plot(args.results_dir, summary)


if __name__ == "__main__":
    main()
