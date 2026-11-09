"""Dispatch training or evaluation, individually or as one evaluation job."""
import argparse
import subprocess
import sys
from pathlib import Path

from jobs import evaluation_jobs, training_jobs


SRC = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("baseline", "train", "evaluate", "evaluate-all", "list"))
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--root", type=Path, default=Path("results/comparison_v1"))
    parser.add_argument("--weights", type=Path, default=Path("weights.pth"))
    parser.add_argument("--examples", type=int, default=512)
    parser.add_argument("--pgd-steps", type=int, default=50)
    parser.add_argument("--eot-samples", type=int, default=16)
    parser.add_argument("--restarts", type=int, default=5)
    args = parser.parse_args()
    if args.stage == "list":
        print(f"baseline=3 train={len(training_jobs())} evaluate={len(evaluation_jobs())}")
        for index, job in enumerate(training_jobs()):
            print(index, job)
        return
    if args.stage == "evaluate-all":
        args.stage = "evaluate"
        jobs = evaluation_jobs()
        for index, job in enumerate(jobs):
            output = evaluation_output(args.root.resolve(), index)
            if output.is_file() and output.with_suffix(".json").is_file():
                print(f"[{index + 1}/{len(jobs)}] already saved: {output.name}", flush=True)
                continue
            print(f"[{index + 1}/{len(jobs)}] {job}", flush=True)
            run_task(args, index, job)
        return
    run_task(args, args.index)


def evaluation_output(root, index):
    return root / "evaluation" / f"{index:04d}.csv"


def run_task(args, index, job=None):
    root = args.root.resolve()
    if args.stage == "baseline":
        seed = (42, 43, 44)[index]
        command = [str(SRC / "main.py"), "--name", "baseline", "--weights", str(args.weights.resolve()),
                   "--output-dir", str(root), "--seed", str(seed),
                   "--head-epochs", "10", "--full-epochs", "40"]
        output = root / "baseline" / f"seed_{seed}" / "model.pt"
    else:
        if job is None:
            job = (training_jobs() if args.stage == "train" else evaluation_jobs())[index]
        baseline = root / "baseline" / f"seed_{job['seed']}"
        common = ["--baseline-dir", str(baseline), "--seed", str(job["seed"])]
        if args.stage == "evaluate":
            model = root / job["name"] / f"seed_{job['seed']}"
            output = evaluation_output(root, index)
            command = [str(SRC / "evaluate_comparison.py"), *common,
                       "--model-dir", str(model), "--output", str(output),
                       "--noise-mode", job["noise_mode"], "--examples", str(args.examples),
                       "--pgd-steps", str(args.pgd_steps), "--eot-samples", str(args.eot_samples),
                       "--restarts", str(args.restarts), "--epsilons", str(job["epsilon"])]
            if job.get("profile_from"):
                command += ["--profile-from", str(root / job["profile_from"])]
            elif job["eval_profile"] is not None:
                command += ["--profile", *map(str, job["eval_profile"])]
        else:
            common += ["--name", job["name"], "--output-dir", str(root)]
            output = root / job["name"] / f"seed_{job['seed']}" / "model.pt"
            if job["kind"] in ("conditional", "film_fixed"):
                command = [str(SRC / "main_conditional.py"), *common, "--max-sigma", "2.5",
                           "--save-final-checkpoint"]
                if job["kind"] == "film_fixed":
                    command += ["--fixed-profile", *map(str, job["profile"])]
            elif job["kind"] == "fixed":
                command = [str(SRC / "main_fixed.py"), *common, "--profile", *map(str, job["profile"]),
                           "--save-final-checkpoint"]
            else:
                command = [str(SRC / "main_adaptive.py"), *common,
                           "--order", *job["order"], "--levels",
                           "0.10", "0.25", "0.40", "0.55", "0.70", "0.85", "1.00",
                           "1.25", "1.50", "1.75", "2.00", "2.25", "2.50",
                           "--consolidation-epochs", "10", "--max-epochs", "100",
                           "--warmup-epochs", "5", "--clean-drop-limit", "0.10",
                           "--skip-test-evaluation"]
    # Deliberately refuse overwriting; new protocol => new output root.
    if output.exists() or output.with_name("result.json").exists():
        raise SystemExit(f"Output already exists: {output}; choose a new --root")
    print(" ".join([sys.executable, *command]), flush=True)
    subprocess.run([sys.executable, *command], check=True)


if __name__ == "__main__":
    main()
