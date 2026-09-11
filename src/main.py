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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=Path("weights.pth"))
    parser.add_argument("--dataset", default="ilee0022/Caltech-256")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--head-epochs", type=int, default=5)
    parser.add_argument("--full-epochs", type=int, default=25)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--full-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--sigma-lateral", type=float, default=0.0)
    parser.add_argument("--sigma-prop", type=float, default=0.0)
    parser.add_argument("--sigma-add", type=float, default=0.0)
    parser.add_argument("--pyramid", action="store_true")
    parser.add_argument("--sigma-pyramid", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--b", type=float, default=1.0)
    parser.add_argument("--sigma-axon", type=float, default=0.0)
    parser.add_argument("--sigma-dendrite", type=float, default=0.0)
    parser.add_argument("--attack-epsilon", type=float, default=8 / 255)
    parser.add_argument("--pgd-alpha", type=float, default=2 / 255)
    parser.add_argument("--pgd-steps", type=int, default=10)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(args):
    model = CORnetRT(
        num_classes=1000,
        sigma_lateral=args.sigma_lateral,
        sigma_prop=args.sigma_prop,
        sigma_add=args.sigma_add,
        pyramid_enabled=args.pyramid,
        sigma_pyramid=args.sigma_pyramid,
        gamma=args.gamma,
        b=args.b,
        sigma_axon=args.sigma_axon,
        sigma_dendrite=args.sigma_dendrite,
    )
    checkpoint = torch.load(args.weights, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("state_dict", checkpoint)
    consume_prefix_in_state_dict_if_present(state_dict, "module.")
    model.load_state_dict(state_dict, strict=True)
    model.decoder.linear = nn.Linear(512, 257)
    return NormalizedModel(model)


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
    )
    torch.save(model.state_dict(), output_dir / "model.pt")
    write_history(output_dir / "history.csv", history)
    clean = evaluate(model, test_loader, device)
    fgsm_result = evaluate(
        model, test_loader, device, fgsm(args.attack_epsilon)
    )
    pgd_result = evaluate(
        model,
        test_loader,
        device,
        pgd(args.attack_epsilon, args.pgd_alpha, args.pgd_steps),
    )
    result = {
        "name": args.name,
        "seed": args.seed,
        "best_validation_accuracy": best_accuracy,
        "epochs_trained": len(history),
        "clean_accuracy": clean["accuracy"],
        "fgsm_accuracy": fgsm_result["accuracy"],
        "pgd_accuracy": pgd_result["accuracy"],
        "sigma_lateral": args.sigma_lateral,
        "sigma_prop": args.sigma_prop,
        "sigma_add": args.sigma_add,
        "pyramid": args.pyramid,
        "sigma_pyramid": args.sigma_pyramid,
        "gamma": args.gamma,
        "b": args.b,
        "sigma_axon": args.sigma_axon,
        "sigma_dendrite": args.sigma_dendrite,
        "attack_epsilon": args.attack_epsilon,
        "pgd_alpha": args.pgd_alpha,
        "pgd_steps": args.pgd_steps,
    }
    with (output_dir / "result.json").open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
