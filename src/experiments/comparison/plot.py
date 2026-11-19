"""Audit completed evaluation files and plot the current comparison snapshot."""
import argparse
import csv
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from jobs import evaluation_jobs


def mean(values):
    return statistics.mean(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "figures")
    args = parser.parse_args()
    jobs = evaluation_jobs()
    records = []
    reference_indices = None
    for index, job in enumerate(jobs):
        path = args.root / "evaluation" / f"{index:04d}.csv"
        if not path.exists():
            print("MISSING", index, job)
            continue
        with path.open(encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))
        assert len(rows) == 1, path
        row = rows[0]
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        assert len(meta["indices"]) == len(set(meta["indices"])) == 512
        if reference_indices is None:
            reference_indices = meta["indices"]
        assert reference_indices == meta["indices"], path
        assert row["model"] == job["name"] and int(row["seed"]) == job["seed"], path
        assert float(row["epsilon_px"]) == job["epsilon"], path
        assert row["noise_mode"] == job["noise_mode"], path
        assert (int(row["examples"]), int(row["pgd_steps"]), int(row["restarts"])) == (512, 50, 5)
        records.append(dict(job, **{key: float(row[key]) * 100 for key in
                                    ("clean_accuracy", "white_pgd_accuracy", "transfer_pgd_accuracy")}))
    print(f"COMPLETE {len(records)}/{len(jobs)}")

    def values(name, label, mode, epsilon, metric, seeds=None):
        return [r[metric] for r in records if r["name"] == name
                and r["profile_label"] == label and r["noise_mode"] == mode
                and r["epsilon"] == epsilon and (seeds is None or r["seed"] in seeds)]

    def stats(name, label, mode, epsilon, metric, seeds=None):
        data = values(name, label, mode, epsilon, metric, seeds)
        return mean(data), statistics.stdev(data) if len(data) > 1 else 0

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sigmas = [0.25, 0.5, 0.85, 1, 2.5]
    families = [("fixed", "Fixed CORnet", "#64748b"),
                ("film_fixed", "Fixed + conditioner", "#d97706"),
                ("conditional", "Conditional", "#2563eb")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    for row_index, mode in enumerate(("off", "on")):
        for column, metric in enumerate(("clean_accuracy", "white_pgd_accuracy")):
            ax = axes[row_index, column]
            for family, label, color in families:
                data = []
                for sigma in sigmas:
                    suffix = f"{round(sigma * 100):03d}"
                    name = "conditional_axon" if family == "conditional" else f"{family}_{suffix}"
                    profile = f"all_{suffix}" if family == "conditional" else "trained"
                    data.append(stats(name, profile, mode, 4, metric))
                ax.errorbar(range(5), [d[0] for d in data], yerr=[d[1] for d in data],
                            marker="o", capsize=3, color=color, label=label)
            ax.set(xticks=range(5), xticklabels=sigmas, xlabel="Common noise intensity sigma",
                   ylabel="Accuracy (%)", title=f"Noise {mode} | " + ("Clean" if column == 0 else "White-box PGD 4/255"))
            ax.grid(alpha=0.2)
            ax.set_ylim(bottom=0)
    axes[0, 0].legend(fontsize=9)
    fig.suptitle("Matched common profiles | mean +/- SD across 3 seeds")
    fig.savefig(args.output_dir / "common_profiles.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")
    for row_index, mode in enumerate(("off", "on")):
        for column, metric in enumerate(("clean_accuracy", "white_pgd_accuracy", "transfer_pgd_accuracy")):
            ax = axes[row_index, column]
            for family, label, color in families:
                name = "conditional_axon" if family == "conditional" else f"{family}_085"
                profile = "all_085" if family == "conditional" else "trained"
                data = [stats(name, profile, mode, e, metric) for e in (2, 4, 8)]
                ax.errorbar((2, 4, 8), [d[0] for d in data], yerr=[d[1] for d in data],
                            marker="o", capsize=3, color=color, label=label)
            ax.set(xticks=(2, 4, 8), xlabel="Attack epsilon (pixel levels / 255)", ylabel="Accuracy (%)",
                   title=f"Noise {mode} | " + ("Clean", "White-box PGD", "Transfer PGD")[column], ylim=(0, 80))
            ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=9)
    fig.suptitle("Profile [0.85, 0.85, 0.85, 0.85] | 3 seeds | same 512 images")
    fig.savefig(args.output_dir / "epsilon_comparison.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    for row_index, mode in enumerate(("off", "on")):
        for column, metric in enumerate(("clean_accuracy", "white_pgd_accuracy")):
            ax = axes[row_index, column]
            for order_index, order in enumerate(("forward", "reverse", "v2_first")):
                available = {r["seed"] for r in records if r["name"] == f"adaptive_{order}"
                             and r["noise_mode"] == mode and r["epsilon"] == 4}
                for offset, name, profile, color, label in (
                    (-0.18, f"adaptive_{order}", "trained", "#64748b", "Adaptive"),
                    (0.18, "conditional_axon", f"matched_adaptive_{order}", "#2563eb", "Conditional, same profile")):
                    avg, sd = stats(name, profile, mode, 4, metric, available)
                    ax.bar(order_index + offset, avg, width=0.34, color=color,
                           label=label if order_index == 0 else None)
                    ax.errorbar(order_index + offset, avg, yerr=sd, color="#111827", capsize=3)
                    if column == 0:
                        print("PAIRED", order, mode, name, "seeds", sorted(available),
                              "clean/white4/transfer4", [round(stats(name, profile, mode, 4, m, available)[0], 3)
                              for m in ("clean_accuracy", "white_pgd_accuracy", "transfer_pgd_accuracy")])
            ax.set(xticks=range(3), xticklabels=("forward", "reverse", "v2_first"),
                   ylabel="Accuracy (%)", title=f"Noise {mode} | " + ("Clean" if column == 0 else "White-box PGD 4/255"))
            ax.grid(axis="y", alpha=0.2)
    axes[0, 0].legend(fontsize=9)
    counts = []
    for mode in ("off", "on"):
        sizes = [len({r["seed"] for r in records if r["name"] == f"adaptive_{order}"
                      and r["noise_mode"] == mode and r["epsilon"] == 4})
                 for order in ("forward", "reverse", "v2_first")]
        counts.append(f"{mode}: n=" + "/".join(map(str, sizes)))
    fig.suptitle("Adaptive profiles | paired available seeds\n" + "; ".join(counts) + " | error bars = SD")
    fig.savefig(args.output_dir / "adaptive_matched.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
