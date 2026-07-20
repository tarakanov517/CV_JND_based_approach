import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim

from create_attacked_datasets import create_adversarial_dataset
from data import create_loaders
from evals import evaluate
from model import Net
from train import train


SEED = 42
LEARNING_RATE = 0.01
MOMENTUM = 0.9
NUM_EPOCHS = 20
LATERAL_SIGMAS = (0.01, 0.03, 0.05, 0.1)
CONTRAST_LEVELS = ((0.02, 0.005), (0.05, 0.01), (0.1, 0.01))
MAGNO_PARVO_LEVELS = ((0.05, 0.025), (0.1, 0.05), (0.15, 0.075))
PYRAMIDAL_SIGMAS = (0.01, 0.03, 0.05)
AXON_SIGMAS = (0.01, 0.03, 0.05, 0.1)
DENDRITE_SIGMAS = (
    (0.025, 0.01, 0.01),
    (0.05, 0.02, 0.02),
    (0.1, 0.03, 0.03),
)
DENDRITE_THETA = (0.5, 1.5, 1.5)
OUTPUT_DIR = Path("experiment_outputs_human_noise")
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
RESULTS_CSV = OUTPUT_DIR / "results.csv"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def experiment(name, noise_type, config):
    return {"name": name, "noise_type": noise_type, "noise_config": config}


def build_experiments():
    experiments = [experiment("baseline", "baseline", {})]

    for sigma in LATERAL_SIGMAS:
        experiments.append(
            experiment(
                f"lateral_{sigma:g}",
                "lateral_inhibition",
                {"lateral_sigma": sigma},
            )
        )

    for sigma_proportional, sigma_additive in CONTRAST_LEVELS:
        experiments.append(
            experiment(
                f"contrast_prop_{sigma_proportional:g}_add_{sigma_additive:g}",
                "contrast_adaptive",
                {
                    "contrast_sigma_proportional": sigma_proportional,
                    "contrast_sigma_additive": sigma_additive,
                },
            )
        )

    for sigma_magno, sigma_parvo in MAGNO_PARVO_LEVELS:
        experiments.append(
            experiment(
                f"magno_{sigma_magno:g}_parvo_{sigma_parvo:g}",
                "magno_parvo",
                {
                    "ratio_magno": 0.1,
                    "sigma_magno": sigma_magno,
                    "fano_magno": 1.5,
                    "sigma_parvo": sigma_parvo,
                    "fano_parvo": 1.1,
                },
            )
        )

    for sigma in PYRAMIDAL_SIGMAS:
        experiments.append(
            experiment(
                f"pyramidal_{sigma:g}",
                "pyramidal",
                {
                    "pyramidal_enabled": True,
                    "pyramidal_sigma": sigma,
                    "pyramidal_gamma": 1.0,
                    "pyramidal_semisaturation": 1.0,
                    "pyramidal_gain_decay": 0.1,
                },
            )
        )

    for sigma in AXON_SIGMAS:
        experiments.append(
            experiment(
                f"axon_{sigma:g}",
                "axon",
                {"axon_sigma": (sigma, sigma, sigma)},
            )
        )

    for sigmas in DENDRITE_SIGMAS:
        experiments.append(
            experiment(
                "dendrite_" + "_".join(f"{value:g}" for value in sigmas),
                "dendrite",
                {
                    "dendrite_theta": DENDRITE_THETA,
                    "dendrite_sigma": sigmas,
                },
            )
        )

    experiments.append(
        experiment(
            "combined_visual_system",
            "combined",
            {
                "lateral_sigma": 0.03,
                "contrast_sigma_proportional": 0.05,
                "contrast_sigma_additive": 0.01,
                "ratio_magno": 0.1,
                "sigma_magno": 0.1,
                "fano_magno": 1.5,
                "sigma_parvo": 0.05,
                "fano_parvo": 1.1,
                "pyramidal_enabled": True,
                "pyramidal_sigma": 0.03,
                "pyramidal_gamma": 1.0,
                "pyramidal_semisaturation": 1.0,
                "pyramidal_gain_decay": 0.1,
                "axon_sigma": (0.03, 0.03, 0.03),
                "dendrite_theta": DENDRITE_THETA,
                "dendrite_sigma": (0.05, 0.02, 0.02),
            },
        )
    )
    return experiments


def read_completed():
    if not RESULTS_CSV.exists():
        return set()
    with RESULTS_CSV.open("r", newline="", encoding="utf-8") as file:
        return {row["name"] for row in csv.DictReader(file)}


def save_result(result):
    rows = []
    if RESULTS_CSV.exists():
        with RESULTS_CSV.open("r", newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
    rows = [row for row in rows if row["name"] != result["name"]]
    rows.append(result)
    temporary_path = RESULTS_CSV.with_suffix(".tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=result.keys())
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(RESULTS_CSV)


def train_or_load(
    current_experiment,
    checkpoint_path,
    train_loader,
    test_loader,
    criterion,
    device,
):
    set_seed(SEED)
    train_loader.generator.manual_seed(SEED)
    model = Net(current_experiment["noise_config"]).to(device)

    if checkpoint_path.exists():
        print("Loading checkpoint", flush=True)
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        if checkpoint.get("experiment") != current_experiment:
            raise RuntimeError(f"Checkpoint configuration mismatch: {checkpoint_path}")
        model.load_state_dict(checkpoint["model_state_dict"])
        return model, checkpoint["training_time"], checkpoint.get("history", [])

    optimizer = optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
    )
    started = time.time()
    history = train(
        net=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        num_epochs=NUM_EPOCHS,
    )
    training_time = time.time() - started
    torch.save(
        {
            "name": current_experiment["name"],
            "experiment": current_experiment,
            "model_state_dict": model.state_dict(),
            "training_time": training_time,
            "history": history,
        },
        checkpoint_path,
    )
    return model, training_time, history


def make_result(
    current_experiment,
    clean,
    fgsm_accuracy,
    pgd_accuracy,
    training_time,
):
    return {
        "name": current_experiment["name"],
        "noise_type": current_experiment["noise_type"],
        "noise_config": json.dumps(
            current_experiment["noise_config"],
            ensure_ascii=False,
            sort_keys=True,
        ),
        "clean_loss": clean[1],
        "clean_accuracy": clean[0],
        "transfer_fgsm_accuracy": fgsm_accuracy,
        "transfer_pgd_accuracy": pgd_accuracy,
        "training_time_seconds": training_time,
    }


def main():
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    train_loader, test_loader = create_loaders(
        train_batch_size=256,
        test_batch_size=512,
        seed=SEED,
    )
    experiments = build_experiments()
    completed = read_completed()
    print(f"Device: {device}", flush=True)
    print(f"Experiments: {len(experiments)}", flush=True)

    baseline_fgsm = None
    baseline_pgd = None

    for index, current_experiment in enumerate(experiments, start=1):
        name = current_experiment["name"]
        if name != "baseline" and name in completed:
            print(f"[{index}/{len(experiments)}] Skip: {name}", flush=True)
            continue

        print(f"[{index}/{len(experiments)}] Experiment: {name}", flush=True)
        model, training_time, _ = train_or_load(
            current_experiment,
            CHECKPOINT_DIR / f"{name}.pt",
            train_loader,
            test_loader,
            criterion,
            device,
        )
        clean = evaluate(model, test_loader, device, criterion)

        if name == "baseline":
            baseline_fgsm = create_adversarial_dataset(
                "fgsm",
                model,
                test_loader,
                criterion,
                device,
                epsilon=(8 / 255) / 0.5,
            )
            baseline_pgd = create_adversarial_dataset(
                "pgd",
                model,
                test_loader,
                criterion,
                device,
                epsilon=(8 / 255) / 0.5,
                alpha=(2 / 255) / 0.5,
                num_iter=10,
            )

        if baseline_fgsm is None or baseline_pgd is None:
            raise RuntimeError("Baseline adversarial datasets were not created")

        fgsm_accuracy, _ = evaluate(model, baseline_fgsm, device)
        pgd_accuracy, _ = evaluate(model, baseline_pgd, device)
        result = make_result(
            current_experiment,
            clean,
            fgsm_accuracy,
            pgd_accuracy,
            training_time,
        )
        save_result(result)
        completed.add(name)
        print(result, flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"Results: {RESULTS_CSV.resolve()}", flush=True)


if __name__ == "__main__":
    main()
