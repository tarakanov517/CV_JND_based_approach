import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from attacks import fgsm, pgd
from cornet_rt import CORnetRT, ConditionalCORnetRT, NormalizedModel
from data import create_loaders


PROFILES = {
    "zero": (0.0, 0.0, 0.0, 0.0),
    "all_025": (0.25, 0.25, 0.25, 0.25),
    "all_050": (0.50, 0.50, 0.50, 0.50),
    "all_085": (0.85, 0.85, 0.85, 0.85),
    "all_100": (1.00, 1.00, 1.00, 1.00),
    "v1_050": (0.50, 0.0, 0.0, 0.0),
    "v2_050": (0.0, 0.50, 0.0, 0.0),
    "v4_050": (0.0, 0.0, 0.50, 0.0),
    "it_050": (0.0, 0.0, 0.0, 0.50),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", default="ilee0022/Caltech-256")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--mc-samples", type=int, default=3)
    parser.add_argument("--eot-samples", type=int, default=4)
    parser.add_argument("--whitebox-examples", type=int, default=512)
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


def load_conditional_model(args, device):
    result_path = args.model_dir / "result.json"
    checkpoint_path = args.model_dir / "model.pt"
    with result_path.open(encoding="utf-8") as file:
        result = json.load(file)
    model = NormalizedModel(
        ConditionalCORnetRT(
            num_classes=257,
            conditioner_hidden=result["conditioner_hidden"],
            max_sigma=result["max_sigma"],
        )
    )
    model.load_state_dict(
        torch.load(checkpoint_path, map_location="cpu", weights_only=True),
        strict=True,
    )
    model = model.to(device).eval()
    model.set_noise_in_eval(True)
    return model, result


def load_baseline(path, device):
    model = NormalizedModel(CORnetRT(num_classes=257))
    model.load_state_dict(
        torch.load(path / "model.pt", map_location="cpu", weights_only=True),
        strict=True,
    )
    return model.to(device).eval()


def averaged_logits(model, inputs, profile, samples):
    logits = model(inputs, profile)
    for _ in range(1, samples):
        logits = logits + model(inputs, profile)
    return logits / samples


def eot_gradient(model, inputs, targets, profile, samples):
    gradient = torch.zeros_like(inputs)
    for _ in range(samples):
        sample = inputs.detach().requires_grad_(True)
        loss = F.cross_entropy(model(sample, profile), targets)
        gradient += torch.autograd.grad(loss, sample)[0]
    return gradient / samples


def eot_fgsm(model, inputs, targets, profile, epsilon, samples):
    gradient = eot_gradient(model, inputs, targets, profile, samples)
    return (inputs + epsilon * gradient.sign()).clamp(0, 1).detach()


def eot_pgd(model, inputs, targets, profile, epsilon, steps, samples):
    adversarial = inputs + torch.empty_like(inputs).uniform_(-epsilon, epsilon)
    adversarial = adversarial.clamp(0, 1).detach()
    alpha = epsilon / 4
    for _ in range(steps):
        gradient = eot_gradient(
            model, adversarial, targets, profile, samples
        )
        adversarial = adversarial + alpha * gradient.sign()
        delta = (adversarial - inputs).clamp(-epsilon, epsilon)
        adversarial = (inputs + delta).clamp(0, 1).detach()
    return adversarial


def evaluate_clean_and_transfer(
    target,
    source,
    loader,
    device,
    profiles,
    epsilons,
    pgd_steps,
    mc_samples,
):
    correct = {
        name: {
            "clean": 0,
            **{
                f"transfer_{attack}_eps{epsilon_key(epsilon)}": 0
                for epsilon in epsilons
                for attack in ("fgsm", "pgd")
            },
        }
        for name in profiles
    }
    total = 0
    for batch_index, (inputs, targets) in enumerate(loader, start=1):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        attacked = {}
        for epsilon_px in epsilons:
            epsilon = epsilon_px / 255
            key = epsilon_key(epsilon_px)
            attacked[("fgsm", key)] = fgsm(epsilon)(source, inputs, targets)
            attacked[("pgd", key)] = pgd(
                epsilon, epsilon / 4, pgd_steps
            )(source, inputs, targets)
        with torch.no_grad():
            for name, values in profiles.items():
                profile = torch.tensor(values, device=device)
                logits = averaged_logits(target, inputs, profile, mc_samples)
                correct[name]["clean"] += (
                    logits.argmax(dim=1) == targets
                ).sum().item()
                for (attack_name, key), attacked_inputs in attacked.items():
                    logits = averaged_logits(
                        target, attacked_inputs, profile, mc_samples
                    )
                    metric = f"transfer_{attack_name}_eps{key}"
                    correct[name][metric] += (
                        logits.argmax(dim=1) == targets
                    ).sum().item()
        total += targets.size(0)
        print(
            f"transfer seed batch={batch_index}/{len(loader)}",
            flush=True,
        )
    return {
        name: {metric: value / total for metric, value in metrics.items()}
        for name, metrics in correct.items()
    }


def evaluate_whitebox(
    model,
    loader,
    device,
    profile,
    epsilons,
    pgd_steps,
    eot_samples,
    mc_samples,
    limit,
):
    correct = {
        f"white_{attack}_eps{epsilon_key(epsilon)}": 0
        for epsilon in epsilons
        for attack in ("fgsm", "pgd")
    }
    total = 0
    profile = torch.tensor(profile, device=device)
    for inputs, targets in loader:
        if total >= limit:
            break
        keep = min(targets.size(0), limit - total)
        inputs = inputs[:keep].to(device, non_blocking=True)
        targets = targets[:keep].to(device, non_blocking=True)
        for epsilon_px in epsilons:
            epsilon = epsilon_px / 255
            key = epsilon_key(epsilon_px)
            fgsm_inputs = eot_fgsm(
                model,
                inputs,
                targets,
                profile,
                epsilon,
                eot_samples,
            )
            pgd_inputs = eot_pgd(
                model,
                inputs,
                targets,
                profile,
                epsilon,
                pgd_steps,
                eot_samples,
            )
            with torch.no_grad():
                fgsm_logits = averaged_logits(
                    model, fgsm_inputs, profile, mc_samples
                )
                pgd_logits = averaged_logits(
                    model, pgd_inputs, profile, mc_samples
                )
            correct[f"white_fgsm_eps{key}"] += (
                fgsm_logits.argmax(dim=1) == targets
            ).sum().item()
            correct[f"white_pgd_eps{key}"] += (
                pgd_logits.argmax(dim=1) == targets
            ).sum().item()
        total += keep
        print(f"white-box examples={total}/{limit}", flush=True)
    return {metric: value / total for metric, value in correct.items()}


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target, training_result = load_conditional_model(args, device)
    source = load_baseline(args.baseline_dir, device)
    _, _, test_loader = create_loaders(
        args.batch_size, args.workers, args.seed, args.dataset
    )
    transfer = evaluate_clean_and_transfer(
        target,
        source,
        test_loader,
        device,
        PROFILES,
        args.attack_epsilons,
        args.pgd_steps,
        args.mc_samples,
    )
    rows = []
    for profile_index, (name, profile) in enumerate(PROFILES.items()):
        set_seed(args.seed * 100 + profile_index)
        print(f"white-box profile={name} values={profile}", flush=True)
        whitebox = evaluate_whitebox(
            target,
            test_loader,
            device,
            profile,
            args.attack_epsilons,
            args.pgd_steps,
            args.eot_samples,
            args.mc_samples,
            args.whitebox_examples,
        )
        row = {
            "profile": name,
            "seed": args.seed,
            "sigma_v1": profile[0],
            "sigma_v2": profile[1],
            "sigma_v4": profile[2],
            "sigma_it": profile[3],
            "clean_accuracy": transfer[name].pop("clean"),
            **transfer[name],
            **whitebox,
            "whitebox_examples": args.whitebox_examples,
            "mc_samples": args.mc_samples,
            "eot_samples": args.eot_samples,
            "pgd_steps": args.pgd_steps,
        }
        rows.append(row)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"evaluation_seed_{args.seed}.csv"
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "seed": args.seed,
        "profiles": PROFILES,
        "attack_epsilons": args.attack_epsilons,
        "whitebox_examples": args.whitebox_examples,
        "mc_samples": args.mc_samples,
        "eot_samples": args.eot_samples,
        "pgd_steps": args.pgd_steps,
        "training": training_result,
    }
    with (args.output_dir / f"evaluation_seed_{args.seed}.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
    print(f"Создан {output}", flush=True)


if __name__ == "__main__":
    main()
