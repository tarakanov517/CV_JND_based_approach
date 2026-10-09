import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

from attacks import evaluate, fgsm, pgd
from cornet_rt import CORnetRT, NormalizedModel
from data import create_loaders
from train import fit


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--noise-kind", choices=("baseline", "axon", "dendrite"))
    parser.add_argument(
        "--noise-block", choices=("none", "V1", "V2", "V4", "IT", "all")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=Path("weights.pth"))
    parser.add_argument("--dataset", default="ilee0022/Caltech-256")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--head-epochs", type=int, default=50)
    parser.add_argument("--full-epochs", type=int, default=100)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--full-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--sigma-axon", type=float, default=0.0)
    parser.add_argument("--sigma-dendrite", type=float, default=0.0)
    parser.add_argument("--attack-epsilons", type=float, nargs="+", default=(2, 4))
    parser.add_argument("--pgd-steps", type=int, default=10)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(args):
    block = "all" if args.noise_block == "none" else args.noise_block
    model = CORnetRT(
        num_classes=1000,
        sigma_axon=args.sigma_axon,
        sigma_dendrite=args.sigma_dendrite,
        parameter_noise_block=block,
    )
    checkpoint = torch.load(args.weights, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("state_dict", checkpoint)
    consume_prefix_in_state_dict_if_present(state_dict, "module.")
    model.load_state_dict(state_dict, strict=True)
    model.decoder.linear = nn.Linear(512, 257)
    return NormalizedModel(model)


def epsilon_key(value):
    return f"{value:g}".replace(".", "p")


def write_history(path, history):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = args.output_dir / args.name / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, validation_loader, test_loader = create_loaders(
        args.batch_size, args.workers, args.seed, args.dataset
    )
    model = build_model(args).to(device)
    history, best_accuracy = fit(
        model,
        train_loader,
        validation_loader,
        device,
        args.head_epochs,
        args.full_epochs,
        args.head_lr,
        args.full_lr,
        args.weight_decay,
        "fixed",
        0,
        1,
    )

    torch.save(model.state_dict(), output_dir / "model.pt")
    write_history(output_dir / "history.csv", history)
    clean = evaluate(model, test_loader, device)

    result = {
        "name": args.name,
        "noise_kind": args.noise_kind,
        "noise_block": args.noise_block,
        "seed": args.seed,
        "best_validation_accuracy": best_accuracy,
        "epochs_trained": len(history),
        "clean_accuracy": clean["accuracy"],
        "sigma_axon": args.sigma_axon,
        "sigma_dendrite": args.sigma_dendrite,
        "attack_epsilons_px": args.attack_epsilons,
        "pgd_steps": args.pgd_steps,
    }

    for epsilon_px in args.attack_epsilons:
        epsilon = epsilon_px / 255
        alpha = epsilon / 4
        key = epsilon_key(epsilon_px)
        fgsm_result = evaluate(model, test_loader, device, fgsm(epsilon))
        pgd_result = evaluate(
            model,
            test_loader,
            device,
            pgd(epsilon, alpha, args.pgd_steps),
        )
        result[f"fgsm_accuracy_eps{key}"] = fgsm_result["accuracy"]
        result[f"pgd_accuracy_eps{key}"] = pgd_result["accuracy"]

    with (output_dir / "result.json").open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
