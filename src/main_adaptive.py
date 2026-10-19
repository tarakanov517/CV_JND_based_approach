import argparse
import copy
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch

from attacks import evaluate, fgsm, pgd
from cornet_rt import CORnetRT, NormalizedModel
from data import create_loaders
from train import set_backbone_trainable, train_epoch


BLOCKS = ("V1", "V2", "V4", "IT")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--order", nargs=4, choices=BLOCKS, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", default="ilee0022/Caltech-256")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--levels", type=float, nargs="+", default=(0.25, 0.40, 0.55, 0.70, 0.85)
    )
    parser.add_argument("--epochs-per-level", type=int, default=4)
    parser.add_argument("--consolidation-epochs", type=int, default=10)
    parser.add_argument("--clean-drop-limit", type=float, default=0.10)
    parser.add_argument("--controller-examples", type=int, default=512)
    parser.add_argument("--controller-epsilon", type=float, default=4.0)
    parser.add_argument("--no-improvement-patience", type=int, default=2)
    parser.add_argument("--min-transfer-improvement", type=float, default=0.002)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
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


def build_model(checkpoint_path, device):
    profile = {block: 0.0 for block in BLOCKS}
    model = NormalizedModel(CORnetRT(num_classes=257, axon_profile=profile))
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    return model.to(device)


def set_axon_profile(model, profile):
    for block, sigma in profile.items():
        module = getattr(model.model, block)
        module.conv_input.sigma_axon = float(sigma)
        module.conv1.sigma_axon = float(sigma)


def build_controller_batches(source, loader, device, epsilon_px, steps, limit):
    attack = pgd(epsilon_px / 255, epsilon_px / 255 / 4, steps)
    batches = []
    collected = 0
    source.eval()
    for inputs, targets in loader:
        if collected >= limit:
            break
        keep = min(targets.size(0), limit - collected)
        inputs = inputs[:keep].to(device, non_blocking=True)
        targets = targets[:keep].to(device, non_blocking=True)
        attacked = attack(source, inputs, targets)
        batches.append((attacked.cpu().half(), targets.cpu()))
        collected += keep
    return batches


def evaluate_cached(model, batches, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in batches:
            inputs = inputs.to(device, dtype=torch.float32, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            correct += (model(inputs).argmax(dim=1) == targets).sum().item()
            total += targets.size(0)
    return correct / total


def train_stage(model, loader, optimizer, device, epochs, start_epoch, label):
    epoch = start_epoch
    for _ in range(epochs):
        epoch += 1
        loss, accuracy = train_epoch(model, loader, optimizer, device)
        print(
            f"adaptive_epoch={epoch} stage={label} "
            f"train_loss={loss:.4f} train_acc={accuracy:.4f}",
            flush=True,
        )
    return epoch


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = args.output_dir / args.name / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    train_loader, validation_loader, test_loader = create_loaders(
        args.batch_size, args.workers, args.seed, args.dataset
    )
    checkpoint_path = args.baseline_dir / "model.pt"
    if not checkpoint_path.exists():
        raise SystemExit(f"Не найден baseline checkpoint: {checkpoint_path}")

    model = build_model(checkpoint_path, device)
    source = build_model(checkpoint_path, device).eval()
    set_backbone_trainable(model, True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    baseline_validation = evaluate(model, validation_loader, device)["accuracy"]
    clean_floor = baseline_validation - args.clean_drop_limit
    controller_batches = build_controller_batches(
        source,
        validation_loader,
        device,
        args.controller_epsilon,
        args.pgd_steps,
        args.controller_examples,
    )
    profile = {block: 0.0 for block in BLOCKS}
    set_axon_profile(model, profile)
    controller_score = evaluate_cached(model, controller_batches, device)
    history = []
    adaptive_epoch = 0

    for block in args.order:
        best_state = copy.deepcopy(model.state_dict())
        best_optimizer_state = copy.deepcopy(optimizer.state_dict())
        best_profile = profile.copy()
        best_score = controller_score
        no_improvement = 0

        for level in args.levels:
            if level <= profile[block]:
                continue
            profile[block] = float(level)
            set_axon_profile(model, profile)
            adaptive_epoch = train_stage(
                model,
                train_loader,
                optimizer,
                device,
                args.epochs_per_level,
                adaptive_epoch,
                f"{block}_{level:.2f}",
            )
            clean_validation = evaluate(model, validation_loader, device)["accuracy"]
            transfer_validation = evaluate_cached(model, controller_batches, device)
            decision = "continue"

            if clean_validation < clean_floor:
                decision = "reject_clean"
                history.append(
                    {
                        "block": block,
                        "level": level,
                        "adaptive_epoch": adaptive_epoch,
                        "clean_validation": clean_validation,
                        "transfer_pgd_validation": transfer_validation,
                        "decision": decision,
                    }
                )
                break

            if transfer_validation > best_score + args.min_transfer_improvement:
                best_score = transfer_validation
                best_state = copy.deepcopy(model.state_dict())
                best_optimizer_state = copy.deepcopy(optimizer.state_dict())
                best_profile = profile.copy()
                no_improvement = 0
                decision = "accept"
            else:
                no_improvement += 1
                if no_improvement >= args.no_improvement_patience:
                    decision = "reject_no_improvement"

            history.append(
                {
                    "block": block,
                    "level": level,
                    "adaptive_epoch": adaptive_epoch,
                    "clean_validation": clean_validation,
                    "transfer_pgd_validation": transfer_validation,
                    "decision": decision,
                }
            )
            print(
                f"block={block} level={level:.2f} clean={clean_validation:.4f} "
                f"transfer_pgd={transfer_validation:.4f} decision={decision}",
                flush=True,
            )
            if decision == "reject_no_improvement":
                break

        model.load_state_dict(best_state)
        optimizer.load_state_dict(best_optimizer_state)
        profile = best_profile
        controller_score = best_score
        set_axon_profile(model, profile)

    before_consolidation_state = copy.deepcopy(model.state_dict())
    before_consolidation_optimizer = copy.deepcopy(optimizer.state_dict())
    before_consolidation_score = controller_score
    before_consolidation_profile = profile.copy()
    adaptive_epoch = train_stage(
        model,
        train_loader,
        optimizer,
        device,
        args.consolidation_epochs,
        adaptive_epoch,
        "consolidation",
    )
    consolidation_clean = evaluate(model, validation_loader, device)["accuracy"]
    consolidation_transfer = evaluate_cached(model, controller_batches, device)
    if (
        consolidation_clean < clean_floor
        or consolidation_transfer + args.min_transfer_improvement
        < before_consolidation_score
    ):
        model.load_state_dict(before_consolidation_state)
        optimizer.load_state_dict(before_consolidation_optimizer)
        profile = before_consolidation_profile
        controller_score = before_consolidation_score
        consolidation_decision = "rollback"
    else:
        controller_score = consolidation_transfer
        consolidation_decision = "accept"

    set_axon_profile(model, profile)
    torch.save(model.state_dict(), run_dir / "model.pt")
    with (run_dir / "adaptive_history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    clean = evaluate(model, test_loader, device)
    result = {
        "name": args.name,
        "noise_kind": "axon",
        "noise_block": "profile",
        "profile_family": "adaptive",
        "seed": args.seed,
        "order": args.order,
        "axon_profile": profile,
        "sigma_axon": 0.0,
        "sigma_dendrite": 0.0,
        "baseline_validation_accuracy": baseline_validation,
        "clean_floor": clean_floor,
        "controller_transfer_pgd_eps4": controller_score,
        "controller_examples": args.controller_examples,
        "adaptive_epochs": adaptive_epoch,
        "consolidation_decision": consolidation_decision,
        "clean_accuracy": clean["accuracy"],
        "attack_epsilons_px": args.attack_epsilons,
        "pgd_steps": args.pgd_steps,
    }
    for epsilon_px in args.attack_epsilons:
        epsilon = epsilon_px / 255
        key = epsilon_key(epsilon_px)
        result[f"fgsm_accuracy_eps{key}"] = evaluate(
            model, test_loader, device, fgsm(epsilon)
        )["accuracy"]
        result[f"pgd_accuracy_eps{key}"] = evaluate(
            model,
            test_loader,
            device,
            pgd(epsilon, epsilon / 4, args.pgd_steps),
        )["accuracy"]

    with (run_dir / "result.json").open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
