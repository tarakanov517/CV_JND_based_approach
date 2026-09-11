import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch

from attacks import fgsm, pgd
from cornet_rt import CORnetRT, NormalizedModel
from data import create_loaders


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("experiments_cornet"))
    parser.add_argument("--dataset", default="ilee0022/Caltech-256")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--attack-epsilon", type=float, default=8 / 255)
    parser.add_argument("--pgd-alpha", type=float, default=2 / 255)
    parser.add_argument("--pgd-steps", type=int, default=10)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model(run_dir, result, device):
    model = CORnetRT(
        num_classes=257,
        sigma_lateral=result["sigma_lateral"],
        sigma_prop=result["sigma_prop"],
        sigma_add=result["sigma_add"],
        pyramid_enabled=result["pyramid"],
        sigma_pyramid=result["sigma_pyramid"],
        gamma=result["gamma"],
        b=result["b"],
        sigma_axon=result["sigma_axon"],
        sigma_dendrite=result["sigma_dendrite"],
    )
    model = NormalizedModel(model)
    state_dict = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    return model


def load_runs(results_dir, seed, device):
    models = {}
    results = {}
    for result_path in sorted(results_dir.glob(f"*/seed_{seed}/result.json")):
        with result_path.open(encoding="utf-8") as file:
            result = json.load(file)
        name = result["name"]
        run_dir = result_path.parent
        models[name] = load_model(run_dir, result, device)
        results[name] = result
    if "baseline" not in models:
        raise SystemExit(f"Не найдена baseline-модель для seed={seed}")
    return models, results


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models, results = load_runs(args.results_dir, args.seed, device)
    _, _, test_loader = create_loaders(
        args.batch_size, args.workers, args.seed, args.dataset
    )
    source = models["baseline"]
    fgsm_attack = fgsm(args.attack_epsilon)
    pgd_attack = pgd(args.attack_epsilon, args.pgd_alpha, args.pgd_steps)
    correct = {
        name: {"fgsm": 0, "pgd": 0}
        for name in models
    }
    total = 0
    for batch_index, (inputs, targets) in enumerate(test_loader, start=1):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        fgsm_inputs = fgsm_attack(source, inputs, targets)
        pgd_inputs = pgd_attack(source, inputs, targets)
        with torch.no_grad():
            for name, model in models.items():
                correct[name]["fgsm"] += (
                    model(fgsm_inputs).argmax(dim=1) == targets
                ).sum().item()
                correct[name]["pgd"] += (
                    model(pgd_inputs).argmax(dim=1) == targets
                ).sum().item()
        total += targets.size(0)
        print(f"seed={args.seed} batch={batch_index}/{len(test_loader)}", flush=True)
    rows = []
    for name in sorted(models):
        rows.append(
            {
                "name": name,
                "seed": args.seed,
                "clean_accuracy": results[name]["clean_accuracy"],
                "baseline_fgsm_accuracy": correct[name]["fgsm"] / total,
                "baseline_pgd_accuracy": correct[name]["pgd"] / total,
            }
        )
    output_path = args.results_dir / f"transfer_seed_{args.seed}.csv"
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Создан {output_path}", flush=True)


if __name__ == "__main__":
    main()
