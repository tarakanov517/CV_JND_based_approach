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
    parser.add_argument("--source-results-dir", type=Path, required=True)
    parser.add_argument("--target-results-dir", type=Path, required=True)
    parser.add_argument("--dataset", default="ilee0022/Caltech-256")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--attack-epsilons", type=float, nargs="+", default=(2, 4, 8))
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
    model = CORnetRT(
        num_classes=257,
        axon_profile=result.get("axon_profile"),
    )
    model = NormalizedModel(model)
    state_dict = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source_dir = args.source_results_dir / "baseline" / f"seed_{args.seed}"
    with (source_dir / "result.json").open(encoding="utf-8") as file:
        source_result = json.load(file)
    source = load_model(source_dir, source_result, device)

    targets = {}
    target_results = {}
    for result_path in sorted(
        args.target_results_dir.glob(f"*/seed_{args.seed}/result.json")
    ):
        with result_path.open(encoding="utf-8") as file:
            result = json.load(file)
        targets[result["name"]] = load_model(result_path.parent, result, device)
        target_results[result["name"]] = result

    if not targets:
        raise SystemExit(f"Не найдены адаптивные модели для seed={args.seed}")

    _, _, test_loader = create_loaders(
        args.batch_size, args.workers, args.seed, args.dataset
    )
    correct = {
        name: {
            epsilon_key(epsilon): {"fgsm": 0, "pgd": 0}
            for epsilon in args.attack_epsilons
        }
        for name in targets
    }
    total = 0

    for batch_index, (inputs, labels) in enumerate(test_loader, start=1):
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        attacked = {}
        for epsilon_px in args.attack_epsilons:
            epsilon = epsilon_px / 255
            key = epsilon_key(epsilon_px)
            attacked[key] = {
                "fgsm": fgsm(epsilon)(source, inputs, labels),
                "pgd": pgd(epsilon, epsilon / 4, args.pgd_steps)(
                    source, inputs, labels
                ),
            }
        with torch.no_grad():
            for name, model in targets.items():
                for key, attacks in attacked.items():
                    for attack_name, attacked_inputs in attacks.items():
                        correct[name][key][attack_name] += (
                            model(attacked_inputs).argmax(dim=1) == labels
                        ).sum().item()
        total += labels.size(0)
        print(f"seed={args.seed} batch={batch_index}/{len(test_loader)}", flush=True)

    rows = []
    for name in sorted(targets):
        result = target_results[name]
        row = {
            "name": name,
            "seed": args.seed,
            "clean_accuracy": result["clean_accuracy"],
        }
        for epsilon_px in args.attack_epsilons:
            key = epsilon_key(epsilon_px)
            row[f"baseline_fgsm_accuracy_eps{key}"] = (
                correct[name][key]["fgsm"] / total
            )
            row[f"baseline_pgd_accuracy_eps{key}"] = correct[name][key]["pgd"] / total
        rows.append(row)

    output = args.target_results_dir / f"transfer_seed_{args.seed}.csv"
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Создан {output}", flush=True)


if __name__ == "__main__":
    main()
