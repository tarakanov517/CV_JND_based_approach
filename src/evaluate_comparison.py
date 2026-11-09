"""Identical subset, checkpoints and attack protocol across all comparison arms."""
import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from cornet_rt import CORnetRT, ConditionalCORnetRT, NormalizedModel
from data import create_loaders
from evaluate_conditional import averaged_logits, eot_pgd, load_baseline, set_seed
from noises import NoisyConv2d


class ProfileModel(torch.nn.Module):
    def __init__(self, network, conditional):
        super().__init__()
        self.network, self.conditional = network, conditional

    def forward(self, inputs, profile=None):
        return self.network(inputs, profile) if self.conditional else self.network(inputs)


def load_target(path, device):
    metadata = json.loads((path / "result.json").read_text(encoding="utf-8"))
    state = torch.load(path / "model.pt", map_location="cpu", weights_only=True)
    conditional = any("conditioner." in key for key in state)
    if conditional:
        network = ConditionalCORnetRT(num_classes=257,
                                      conditioner_hidden=metadata["conditioner_hidden"],
                                      max_sigma=metadata["max_sigma"])
        profile = metadata.get("fixed_profile") or [0.0] * 4
    else:
        profile_dict = metadata.get("axon_profile", {})
        profile = [profile_dict.get(b, metadata.get("sigma_axon", 0.0))
                   for b in ("V1", "V2", "V4", "IT")]
        network = CORnetRT(num_classes=257, axon_profile=dict(zip(
            ("V1", "V2", "V4", "IT"), profile)))
    network = NormalizedModel(network)
    network.load_state_dict(state, strict=True)
    return ProfileModel(network, conditional).to(device).eval(), profile


def best_pgd(model, inputs, targets, profile, epsilon, steps, eot, samples, restarts):
    # Include the original point, so PGD cannot select a lower-loss candidate.
    best = inputs.detach().clone()
    with torch.no_grad():
        best_loss = F.cross_entropy(averaged_logits(model, best, profile, samples),
                                    targets, reduction="none")
    for _ in range(restarts):
        candidate = eot_pgd(model, inputs, targets, profile, epsilon, steps, eot)
        with torch.no_grad():
            loss = F.cross_entropy(averaged_logits(model, candidate, profile, samples),
                                   targets, reduction="none")
        take = loss > best_loss
        best[take] = candidate[take]
        best_loss = torch.maximum(best_loss, loss)
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--profile", type=float, nargs=4)
    parser.add_argument("--profile-from", type=Path)
    parser.add_argument("--noise-mode", choices=("on", "off"), required=True)
    parser.add_argument("--dataset", default="ilee0022/Caltech-256")
    parser.add_argument("--examples", type=int, default=512)
    parser.add_argument("--subset-seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--epsilons", type=float, nargs="+", default=[2, 4, 8])
    parser.add_argument("--pgd-steps", type=int, default=50)
    parser.add_argument("--restarts", type=int, default=5)
    parser.add_argument("--eot-samples", type=int, default=16)
    parser.add_argument("--mc-samples", type=int, default=3)
    args = parser.parse_args()
    if min(args.examples, args.pgd_steps, args.restarts, args.eot_samples,
           args.mc_samples) < 1 or min(args.epsilons) <= 0:
        parser.error("counts and epsilons must be positive")
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, profile = load_target(args.model_dir, device)
    if args.profile is not None:
        if not model.conditional and args.profile != profile:
            parser.error("Plain fixed models must be evaluated at their trained profile")
        profile = args.profile
    if args.profile_from:
        if not model.conditional:
            parser.error("--profile-from requires a conditional model")
        meta = json.loads(args.profile_from.read_text(encoding="utf-8"))
        profile = [meta["axon_profile"][b] for b in ("V1", "V2", "V4", "IT")]
    if model.conditional and max(profile) > model.network.model.conditioner.max_sigma:
        parser.error("Adaptive profile exceeds the conditional training range")
    enabled = args.noise_mode == "on"
    if model.conditional:
        model.network.set_noise_in_eval(enabled)
    else:
        for module in model.modules():
            if isinstance(module, NoisyConv2d):
                module.noise_in_eval = enabled
    # Disabling noise leaves conditioner/profile intact; it is not a zero profile.
    source = ProfileModel(load_baseline(args.baseline_dir, device), False)
    _, _, full_loader = create_loaders(args.batch_size, args.workers, args.seed, args.dataset)
    indices = torch.randperm(len(full_loader.dataset), generator=
                             torch.Generator().manual_seed(args.subset_seed))[:args.examples].tolist()
    loader = DataLoader(Subset(full_loader.dataset, indices), batch_size=args.batch_size,
                        shuffle=False, num_workers=args.workers)
    profile = torch.tensor(profile, device=device)
    stochastic = enabled and bool(torch.any(profile != 0))
    samples = args.mc_samples if stochastic else 1
    eot = args.eot_samples if stochastic else 1
    rows = []
    for epsilon_px in args.epsilons:
        # Reset for each epsilon: the source attack is identical across targets.
        set_seed(args.seed)
        clean_correct = white_correct = transfer_correct = total = 0
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            epsilon = epsilon_px / 255
            # Fork RNG to keep the transfer examples independent of target randomness.
            with torch.random.fork_rng(devices=[device.index or 0] if device.type == "cuda" else []):
                set_seed(args.seed * 100000 + total)
                transfer = best_pgd(source, inputs, targets, None, epsilon,
                                    args.pgd_steps, 1, 1, args.restarts)
            white = best_pgd(model, inputs, targets, profile, epsilon,
                             args.pgd_steps, eot, samples, args.restarts)
            with torch.no_grad():
                clean_correct += (averaged_logits(model, inputs, profile, samples).argmax(1) == targets).sum().item()
                white_correct += (averaged_logits(model, white, profile, samples).argmax(1) == targets).sum().item()
                transfer_correct += (averaged_logits(model, transfer, profile, samples).argmax(1) == targets).sum().item()
            total += targets.numel()
            print(f"eps={epsilon_px} examples={total}/{len(indices)}", flush=True)
        rows.append(dict(model=args.model_dir.parent.name, seed=args.seed,
                         profile=json.dumps(profile.tolist()), noise_mode=args.noise_mode,
                         epsilon_px=epsilon_px, examples=total, clean_accuracy=clean_correct / total,
                         white_pgd_accuracy=white_correct / total,
                         transfer_pgd_accuracy=transfer_correct / total,
                         pgd_steps=args.pgd_steps, restarts=args.restarts,
                         eot_samples=eot, mc_samples=samples, subset_seed=args.subset_seed))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pending_output = args.output.with_suffix(".csv.tmp")
    with pending_output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    args.output.with_suffix(".json").write_text(json.dumps(dict(
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        indices=indices, torch_version=torch.__version__), indent=2), encoding="utf-8")
    # Publish the CSV only after its metadata is ready. A retry can replace a partial temporary file.
    pending_output.replace(args.output)


if __name__ == "__main__":
    main()
