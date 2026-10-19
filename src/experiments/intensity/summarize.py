import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


METRICS = (
    "clean_accuracy",
    "white_fgsm_eps2",
    "white_pgd_eps2",
    "transfer_fgsm_eps2",
    "transfer_pgd_eps2",
    "white_fgsm_eps4",
    "white_pgd_eps4",
    "transfer_fgsm_eps4",
    "transfer_pgd_eps4",
    "white_fgsm_eps8",
    "white_pgd_eps8",
    "transfer_fgsm_eps8",
    "transfer_pgd_eps8",
)


def intensity(result):
    if result["noise_kind"] == "axon":
        return float(result["sigma_axon"])
    if result["noise_kind"] == "dendrite":
        return float(result["sigma_dendrite"])
    return 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    args = parser.parse_args()

    transfer = {}
    for path in args.results.glob("transfer_seed_*.csv"):
        with path.open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                transfer[(row["name"], int(row["seed"]))] = row

    rows = []
    for path in args.results.glob("*/seed_*/result.json"):
        with path.open(encoding="utf-8") as file:
            result = json.load(file)
        transfer_row = transfer.get((result["name"], int(result["seed"])), {})
        row = {
            "name": result["name"],
            "seed": result["seed"],
            "noise_kind": result["noise_kind"],
            "noise_block": result["noise_block"],
            "intensity": intensity(result),
            "sigma_axon": result["sigma_axon"],
            "sigma_dendrite": result["sigma_dendrite"],
            "clean_accuracy": result["clean_accuracy"],
        }
        for epsilon in (2, 4, 8):
            row[f"white_fgsm_eps{epsilon}"] = result[
                f"fgsm_accuracy_eps{epsilon}"
            ]
            row[f"white_pgd_eps{epsilon}"] = result[f"pgd_accuracy_eps{epsilon}"]
            row[f"transfer_fgsm_eps{epsilon}"] = transfer_row.get(
                f"baseline_fgsm_accuracy_eps{epsilon}", ""
            )
            row[f"transfer_pgd_eps{epsilon}"] = transfer_row.get(
                f"baseline_pgd_accuracy_eps{epsilon}", ""
            )
        rows.append(row)

    if not rows:
        raise SystemExit("Файлы result.json не найдены")

    rows.sort(
        key=lambda row: (
            row["noise_kind"],
            row["noise_block"],
            float(row["intensity"]),
            int(row["seed"]),
        )
    )
    with (args.results / "intensity_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    groups = defaultdict(list)
    for row in rows:
        key = (row["noise_kind"], row["noise_block"], float(row["intensity"]))
        groups[key].append(row)

    summary = []
    for (kind, block, level), group in sorted(groups.items()):
        item = {
            "noise_kind": kind,
            "noise_block": block,
            "intensity": level,
            "seeds": len(group),
        }
        for metric in METRICS:
            values = np.array(
                [float(row[metric]) for row in group if row[metric] != ""],
                dtype=float,
            )
            item[f"{metric}_mean"] = values.mean() if len(values) else ""
            item[f"{metric}_std"] = (
                values.std(ddof=1)
                if len(values) > 1
                else 0.0 if len(values) else ""
            )
        summary.append(item)

    with (args.results / "intensity_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)

    print(f"Собрано запусков: {len(rows)}")
    print("Созданы intensity_results.csv и intensity_summary.csv")


if __name__ == "__main__":
    main()
