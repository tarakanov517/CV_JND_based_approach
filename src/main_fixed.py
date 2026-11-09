"""Plain CORnet-RT specialist, initialized from a shared Caltech baseline."""
import argparse
import copy
import json
from pathlib import Path

import torch

from attacks import evaluate
from data import create_loaders
from main_adaptive import BLOCKS, build_model, set_axon_profile
from main_conditional import set_seed, write_history
from train import train_epoch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--profile", type=float, nargs=4, required=True)
    parser.add_argument("--dataset", default="ilee0022/Caltech-256")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--ramp", type=int, default=25)
    parser.add_argument("--save-final-checkpoint", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or min(args.profile) < 0:
        parser.error("epochs must be positive and profile nonnegative")
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.baseline_dir / "model.pt", device)
    train, validation, _ = create_loaders(args.batch_size, args.workers, args.seed, args.dataset)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    best_score, best_state, history = -1, None, []
    for epoch in range(1, args.epochs + 1):
        scale = min(max((epoch - args.warmup) / max(args.ramp, 1), 0), 1)
        set_axon_profile(model, dict(zip(BLOCKS, (v * scale for v in args.profile))))
        loss, accuracy = train_epoch(model, train, optimizer, device)
        score = evaluate(model, validation, device)["accuracy"]
        history.append(dict(epoch=epoch, train_loss=loss, train_accuracy=accuracy,
                            validation_accuracy=score, noise_scale=scale))
        print(history[-1], flush=True)
        if score > best_score or args.save_final_checkpoint:
            best_score, best_state = score, copy.deepcopy(model.state_dict())
        scheduler.step()
    output = args.output_dir / args.name / f"seed_{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, output / "model.pt")
    write_history(output / "history.csv", history)
    result = dict(name=args.name, seed=args.seed, model_kind="plain_fixed",
                  axon_profile=dict(zip(BLOCKS, args.profile)),
                  best_validation_accuracy=best_score,
                  arguments={k: str(v) if isinstance(v, Path) else v
                             for k, v in vars(args).items()})
    (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
