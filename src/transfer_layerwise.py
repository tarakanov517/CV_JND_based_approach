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
    parser.add_argument("--results-dir", type=Path, default=Path("layerwise_results"))
    parser.add_argument("--dataset", default="ilee0022/Caltech-256")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--attack-epsilons", type=float, nargs="+", default=(2, 4))
    parser.add_argument("--pgd-steps", type=int, default=10)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def epsilon_key(value):
    return f"{value:g}".replace(".", "p")


def load_model(run_dir, result, device):
    block = "all" if result["noise_block"] == "none" else result["noise_block"]
    model = CORnetRT(
        num_classes=257,
        sigma_axon=result["sigma_axon"],
        sigma_dendrite=result["sigma_dendrite"],
        parameter_noise_block=block,
    )
    model = NormalizedModel(model)
    state_dict = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval()


def load_runs(results_dir, seed, device):
    models = {}
    results = {}
    for result_path in sorted(results_dir.glob(f"*/seed_{seed}/result.json")):
        with result_path.open(encoding="utf-8") as file:
            result = json.load(file)
        models[result["name"]] = load_model(result_path.parent, result, device)
        results[result["name"]] = result
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
    correct = {
        name: {
            epsilon_key(epsilon): {"fgsm": 0, "pgd": 0}
            for epsilon in args.attack_epsilons
        }
        for name in models
    }
    total = 0

    for batch_index, (inputs, targets) in enumerate(test_loader, start=1):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        adversarial = {}
        for epsilon_px in args.attack_epsilons:
            epsilon = epsilon_px / 255
            key = epsilon_key(epsilon_px)
            adversarial[key] = {
                "fgsm": fgsm(epsilon)(source, inputs, targets),
                "pgd": pgd(epsilon, epsilon / 4, args.pgd_steps)(
                    source, inputs, targets
                ),
            }
        with torch.no_grad():
            for name, model in models.items():
                for key, attacks in adversarial.items():
                    for attack_name, attacked_inputs in attacks.items():
                        correct[name][key][attack_name] += (
                            model(attacked_inputs).argmax(dim=1) == targets
                        ).sum().item()
        total += targets.size(0)
        print(f"seed={args.seed} batch={batch_index}/{len(test_loader)}", flush=True)

    rows = []
    for name in sorted(models):
        result = results[name]
        row = {
            "name": name,
            "seed": args.seed,
            "noise_kind": result["noise_kind"],
            "noise_block": result["noise_block"],
            "clean_accuracy": result["clean_accuracy"],
        }
        for epsilon_px in args.attack_epsilons:
            key = epsilon_key(epsilon_px)
            row[f"baseline_fgsm_accuracy_eps{key}"] = (
                correct[name][key]["fgsm"] / total
            )
            row[f"baseline_pgd_accuracy_eps{key}"] = (
                correct[name][key]["pgd"] / total
            )
        rows.append(row)

    output_path = args.results_dir / f"transfer_seed_{args.seed}.csv"
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Создан {output_path}", flush=True)


if __name__ == "__main__":
    main()
