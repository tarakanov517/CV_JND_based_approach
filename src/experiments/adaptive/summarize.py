import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


BLOCKS = ("V1", "V2", "V4", "IT")
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
        profile = result["axon_profile"]
        transfer_row = transfer.get((result["name"], int(result["seed"])), {})
        row = {
            "name": result["name"],
            "seed": result["seed"],
            "order": ">".join(result["order"]),
            "sigma_v1": profile["V1"],
            "sigma_v2": profile["V2"],
            "sigma_v4": profile["V4"],
            "sigma_it": profile["IT"],
            "total_sigma": sum(profile.values()),
            "adaptive_epochs": result["adaptive_epochs"],
            "clean_accuracy": result["clean_accuracy"],
        }
        for epsilon in (2, 4, 8):
            row[f"white_fgsm_eps{epsilon}"] = result[f"fgsm_accuracy_eps{epsilon}"]
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

    rows.sort(key=lambda row: (row["name"], int(row["seed"])))
    with (args.results / "adaptive_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    groups = defaultdict(list)
    for row in rows:
        groups[row["name"]].append(row)

    summary = []
    for name, group in sorted(groups.items()):
        item = {
            "name": name,
            "order": group[0]["order"],
            "seeds": len(group),
        }
        for block in BLOCKS:
            key = f"sigma_{block.lower()}"
            values = np.array([float(row[key]) for row in group])
            item[f"{key}_mean"] = values.mean()
            item[f"{key}_std"] = values.std(ddof=1) if len(values) > 1 else 0.0
        for metric in METRICS:
            values = np.array(
                [float(row[metric]) for row in group if row[metric] != ""],
                dtype=float,
            )
            item[f"{metric}_mean"] = values.mean() if len(values) else ""
            item[f"{metric}_std"] = values.std(ddof=1) if len(values) > 1 else 0.0
        summary.append(item)

    with (args.results / "adaptive_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)

    print(f"Собрано запусков: {len(rows)}")
    print("Созданы adaptive_results.csv и adaptive_summary.csv")


if __name__ == "__main__":
    main()
