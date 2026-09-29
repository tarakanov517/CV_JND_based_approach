import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


METRICS = (
    "clean_accuracy",
    "white_fgsm_accuracy",
    "white_pgd_accuracy",
    "transfer_fgsm_accuracy",
    "transfer_pgd_accuracy",
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
        key = (result["name"], int(result["seed"]))
        transfer_row = transfer.get(key, {})
        rows.append(
            {
                "name": result["name"],
                "seed": result["seed"],
                "noise_schedule": result["noise_schedule"],
                "sigma_axon": result["sigma_axon"],
                "sigma_dendrite": result["sigma_dendrite"],
                "clean_accuracy": result["clean_accuracy"],
                "white_fgsm_accuracy": result["fgsm_accuracy"],
                "white_pgd_accuracy": result["pgd_accuracy"],
                "transfer_fgsm_accuracy": transfer_row.get(
                    "baseline_fgsm_accuracy", ""
                ),
                "transfer_pgd_accuracy": transfer_row.get(
                    "baseline_pgd_accuracy", ""
                ),
            }
        )

    if not rows:
        raise SystemExit("Файлы result.json не найдены")

    with (args.results / "curriculum_results.csv").open(
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
            "noise_schedule": group[0]["noise_schedule"],
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

    with (args.results / "curriculum_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)

    print(f"Собрано запусков: {len(rows)}")
    print(f"Созданы curriculum_results.csv и curriculum_summary.csv")


if __name__ == "__main__":
    main()
