"""Aggregate only matching protocols; keep one row per epsilon and profile."""
import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    rows = []
    for path in sorted((args.root / "evaluation").glob("*.csv")):
        with path.open(encoding="utf-8", newline="") as file:
            rows.extend(csv.DictReader(file))
    if not rows:
        raise SystemExit("No evaluation CSVs")
    keys = ("model", "profile", "noise_mode", "epsilon_px", "examples", "pgd_steps",
            "restarts", "eot_samples", "mc_samples", "subset_seed")
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    summary = []
    for key, group in sorted(groups.items()):
        seeds = [r["seed"] for r in group]
        if len(set(seeds)) != len(seeds):
            raise SystemExit(f"Duplicate seed in group {key}")
        item = dict(zip(keys, key), seeds=len(seeds))
        for metric in ("clean_accuracy", "white_pgd_accuracy", "transfer_pgd_accuracy"):
            values = [100 * float(r[metric]) for r in group]
            item[metric + "_mean_pct"] = statistics.mean(values)
            item[metric + "_std_pct"] = statistics.stdev(values) if len(values) > 1 else ""
        summary.append(item)
    for filename, content in (("results.csv", rows), ("summary.csv", summary)):
        with (args.root / filename).open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=content[0])
            writer.writeheader()
            writer.writerows(content)
    print(f"{len(rows)} rows; {len(summary)} groups. Adaptive profiles that differ by seed stay separate.")


if __name__ == "__main__":
    main()
