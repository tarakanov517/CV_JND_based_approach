import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    rows = []
    for path in args.results.glob("*/seed_*/result.json"):
        with path.open(encoding="utf-8") as file:
            rows.append(json.load(file))
    if not rows:
        raise SystemExit("Файлы result.json не найдены")
    fields = list(rows[0].keys())
    with (args.results / "results.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (row["name"], row["seed"])))
    groups = defaultdict(list)
    for row in rows:
        groups[row["name"]].append(row)
    summary = []
    for name, group in sorted(groups.items()):
        item = {"name": name, "seeds": len(group)}
        for metric in ("clean_accuracy", "fgsm_accuracy", "pgd_accuracy"):
            values = np.array([row[metric] for row in group], dtype=float)
            item[f"{metric}_mean"] = values.mean()
            item[f"{metric}_std"] = values.std(ddof=1) if len(values) > 1 else 0.0
        summary.append(item)
    with (args.results / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)
    print(f"Собрано запусков: {len(rows)}")
    print(f"Созданы {args.results / 'results.csv'} и {args.results / 'summary.csv'}")


if __name__ == "__main__":
    main()
