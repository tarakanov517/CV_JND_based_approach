import argparse
import copy
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from cornet_rt import ConditionalCORnetRT, NormalizedModel
from data import create_loaders


BLOCKS = ("V1", "V2", "V4", "IT")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="conditional_axon")
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", default="ilee0022/Caltech-256")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-sigma", type=float, default=1.0)
    parser.add_argument("--anchor-sigma", type=float, default=0.85)
    parser.add_argument("--conditioner-hidden", type=int, default=128)
    parser.add_argument("--noise-warmup-epochs", type=int, default=5)
    parser.add_argument("--noise-ramp-epochs", type=int, default=25)
    parser.add_argument("--zero-probability", type=float, default=0.20)
    parser.add_argument("--shared-probability", type=float, default=0.30)
    parser.add_argument("--single-probability", type=float, default=0.30)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(args, checkpoint_path):
    model = NormalizedModel(
        ConditionalCORnetRT(
            num_classes=257,
            conditioner_hidden=args.conditioner_hidden,
            max_sigma=args.max_sigma,
        )
    )
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    allowed_missing = tuple(
        key for key in missing if key.startswith("model.conditioner.")
    )
    if len(allowed_missing) != len(missing) or unexpected:
        raise RuntimeError(
            f"Несовместимый baseline: missing={missing}, unexpected={unexpected}"
        )
    return model


def current_max_sigma(args, epoch):
    if epoch <= args.noise_warmup_epochs:
        return 0.0
    progress = (epoch - args.noise_warmup_epochs) / max(
        args.noise_ramp_epochs, 1
    )
    return args.max_sigma * min(progress, 1.0)


def sample_profile(args, maximum, device):
    draw = torch.rand((), device=device).item()
    zero_end = args.zero_probability
    shared_end = zero_end + args.shared_probability
    single_end = shared_end + args.single_probability
    if draw < zero_end or maximum == 0:
        return torch.zeros(4, device=device), "zero"
    if draw < shared_end:
        value = torch.rand((), device=device) * maximum
        return value.repeat(4), "shared"
    if draw < single_end:
        profile = torch.zeros(4, device=device)
        index = torch.randint(0, 4, (), device=device)
        profile[index] = torch.rand((), device=device) * maximum
        return profile, "single"
    return torch.rand(4, device=device) * maximum, "independent"


def train_epoch(model, loader, optimizer, device, args, maximum):
    model.train()
    model.set_noise_in_eval(False)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    profile_sum = torch.zeros(4, device=device)
    mode_counts = {"zero": 0, "shared": 0, "single": 0, "independent": 0}
    batches = 0
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        profile, mode = sample_profile(args, maximum, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs, profile)
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * targets.size(0)
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_examples += targets.size(0)
        profile_sum += profile.detach()
        mode_counts[mode] += 1
        batches += 1
    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
        "mean_profile": (profile_sum / batches).tolist(),
        "mode_counts": mode_counts,
    }


def evaluate_profile(model, loader, device, profile, mc_samples=1):
    model.eval()
    model.set_noise_in_eval(True)
    profile = torch.tensor(profile, device=device, dtype=torch.float32)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = sum(model(inputs, profile) for _ in range(mc_samples))
            logits = logits / mc_samples
            total_loss += F.cross_entropy(
                logits, targets, reduction="sum"
            ).item()
            total_correct += (logits.argmax(dim=1) == targets).sum().item()
            total_examples += targets.size(0)
    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }


def write_history(path, history):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)


def main():
    args = parse_args()
    set_seed(args.seed)
    probability_sum = (
        args.zero_probability
        + args.shared_probability
        + args.single_probability
    )
    if probability_sum > 1.0:
        raise SystemExit("Сумма вероятностей zero/shared/single не должна превышать 1")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = args.output_dir / args.name / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.baseline_dir / "model.pt"
    if not checkpoint_path.exists():
        raise SystemExit(f"Не найден baseline checkpoint: {checkpoint_path}")

    train_loader, validation_loader, _ = create_loaders(
        args.batch_size, args.workers, args.seed, args.dataset
    )
    model = build_model(args, checkpoint_path).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs
    )
    zero_profile = [0.0, 0.0, 0.0, 0.0]
    anchor_profile = [args.anchor_sigma] * 4
    best_score = -1.0
    best_state = None
    history = []

    for epoch in range(1, args.epochs + 1):
        maximum = current_max_sigma(args, epoch)
        training = train_epoch(
            model, train_loader, optimizer, device, args, maximum
        )
        validation_zero = evaluate_profile(
            model, validation_loader, device, zero_profile
        )
        validation_anchor = evaluate_profile(
            model, validation_loader, device, anchor_profile
        )
        scheduler.step()
        selection_score = (
            validation_zero["accuracy"] + validation_anchor["accuracy"]
        ) / 2
        row = {
            "epoch": epoch,
            "train_loss": training["loss"],
            "train_accuracy": training["accuracy"],
            "max_sampled_sigma": maximum,
            "mean_sigma_v1": training["mean_profile"][0],
            "mean_sigma_v2": training["mean_profile"][1],
            "mean_sigma_v4": training["mean_profile"][2],
            "mean_sigma_it": training["mean_profile"][3],
            "validation_zero_accuracy": validation_zero["accuracy"],
            "validation_anchor_accuracy": validation_anchor["accuracy"],
            "selection_score": selection_score,
            "zero_batches": training["mode_counts"]["zero"],
            "shared_batches": training["mode_counts"]["shared"],
            "single_batches": training["mode_counts"]["single"],
            "independent_batches": training["mode_counts"]["independent"],
        }
        history.append(row)
        print(
            f"epoch={epoch} max_sigma={maximum:.3f} "
            f"train_acc={training['accuracy']:.4f} "
            f"val_zero={validation_zero['accuracy']:.4f} "
            f"val_anchor={validation_anchor['accuracy']:.4f}",
            flush=True,
        )
        if selection_score > best_score:
            best_score = selection_score
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), run_dir / "model.pt")
    write_history(run_dir / "history.csv", history)
    final_zero = evaluate_profile(model, validation_loader, device, zero_profile, 3)
    final_anchor = evaluate_profile(
        model, validation_loader, device, anchor_profile, 3
    )
    result = {
        "name": args.name,
        "seed": args.seed,
        "model_kind": "conditional_axon",
        "blocks": BLOCKS,
        "epochs": args.epochs,
        "max_sigma": args.max_sigma,
        "anchor_sigma": args.anchor_sigma,
        "conditioner_hidden": args.conditioner_hidden,
        "noise_warmup_epochs": args.noise_warmup_epochs,
        "noise_ramp_epochs": args.noise_ramp_epochs,
        "sampling_probabilities": {
            "zero": args.zero_probability,
            "shared": args.shared_probability,
            "single": args.single_probability,
            "independent": 1.0 - probability_sum,
        },
        "best_selection_score": best_score,
        "validation_zero_accuracy": final_zero["accuracy"],
        "validation_anchor_accuracy": final_anchor["accuracy"],
    }
    with (run_dir / "result.json").open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
