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
)


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
            "sigma_axon": result["sigma_axon"],
            "sigma_dendrite": result["sigma_dendrite"],
            "clean_accuracy": result["clean_accuracy"],
        }
        for epsilon in (2, 4):
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

    with (args.results / "layerwise_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (row["name"], row["seed"])))

    groups = defaultdict(list)
    for row in rows:
        groups[row["name"]].append(row)

    summary = []
    for name, group in sorted(groups.items()):
        item = {
            "name": name,
            "seeds": len(group),
            "noise_kind": group[0]["noise_kind"],
            "noise_block": group[0]["noise_block"],
            "sigma_axon": group[0]["sigma_axon"],
            "sigma_dendrite": group[0]["sigma_dendrite"],
        }
        for metric in METRICS:
            values = np.array(
                [float(row[metric]) for row in group if row[metric] != ""],
                dtype=float,
            )
            item[f"{metric}_mean"] = values.mean() if len(values) else ""
            item[f"{metric}_std"] = (
                values.std(ddof=1) if len(values) > 1 else 0.0 if len(values) else ""
            )
        summary.append(item)

    with (args.results / "layerwise_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)

    print(f"Собрано запусков: {len(rows)}")
    print("Созданы layerwise_results.csv и layerwise_summary.csv")


if __name__ == "__main__":
    main()
