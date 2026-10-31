import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


IDENTIFIERS = {
    "profile",
    "seed",
    "sigma_v1",
    "sigma_v2",
    "sigma_v4",
    "sigma_it",
    "whitebox_examples",
    "mc_samples",
    "eot_samples",
    "pgd_steps",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.results.glob("evaluation_seed_*.csv")):
        with path.open(newline="", encoding="utf-8") as file:
            rows.extend(csv.DictReader(file))
    if not rows:
        raise SystemExit("Не найдены evaluation_seed_*.csv")
    rows.sort(key=lambda row: (row["profile"], int(row["seed"])))
    with (args.results / "conditional_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    groups = defaultdict(list)
    for row in rows:
        groups[row["profile"]].append(row)
    metrics = [key for key in rows[0] if key not in IDENTIFIERS]
    summary = []
    for profile, group in sorted(groups.items()):
        item = {
            "profile": profile,
            "seeds": len(group),
            "sigma_v1": group[0]["sigma_v1"],
            "sigma_v2": group[0]["sigma_v2"],
            "sigma_v4": group[0]["sigma_v4"],
            "sigma_it": group[0]["sigma_it"],
        }
        for metric in metrics:
            values = np.array([float(row[metric]) for row in group])
            item[f"{metric}_mean"] = values.mean()
            item[f"{metric}_std"] = (
                values.std(ddof=1) if len(values) > 1 else 0.0
            )
        summary.append(item)
    with (args.results / "conditional_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)
    print(f"Собрано строк: {len(rows)}")
    print("Созданы conditional_results.csv и conditional_summary.csv")


if __name__ == "__main__":
    main()
