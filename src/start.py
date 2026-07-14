import csv
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
LR = 0.01
MOMENTUM = 0.9
NUM_EPOCHS = 20

ACTIVATION_SIGMA_GRID = [
    0.0,
    0.0025,
    0.005,
    0.01,
    0.015,
    0.02,
    0.03,
    0.04,
    0.05,
    0.07,
    0.10,
    0.15,
]

PARAMETER_OMEGA_GRID = [
    0.0,
    1e-5,
    3e-5,
    1e-4,
    2e-4,
    3e-4,
    5e-4,
    1e-3,
    2e-3,
    3e-3,
    5e-3,
    1e-2,
]

RUN_ACTIVATION_EXPERIMENTS = True
RUN_PARAMETER_EXPERIMENTS = True

OUTPUT_DIR = Path("experiment_outputs_gaussian_grid")
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
RESULTS_CSV = OUTPUT_DIR / "results.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def build_experiments():
    experiments = [
        {
            "name": "baseline",
            "noise_type": "baseline",
            "group": "none",
            "value": 0.0,
            "activation_sigmas": (0.0, 0.0, 0.0),
            "parameter_omegas": (0.0, 0.0, 0.0),
        }
    ]

    groups = (
        ("early", 0),
        ("late", 1),
        ("classifier", 2),
    )

    if RUN_ACTIVATION_EXPERIMENTS:
        for sigma in ACTIVATION_SIGMA_GRID:
            if sigma == 0:
                continue

            for group_name, group_index in groups:
                activation_sigmas = [0.0, 0.0, 0.0]
                activation_sigmas[group_index] = sigma

                experiments.append(
                    {
                        "name": (
                            f"activation_"
                            f"{group_name}_"
                            f"{sigma:g}"
                        ),
                        "noise_type": "activation",
                        "group": group_name,
                        "value": sigma,
                        "activation_sigmas": tuple(
                            activation_sigmas
                        ),
                        "parameter_omegas": (
                            0.0,
                            0.0,
                            0.0,
                        ),
                    }
                )

    if RUN_PARAMETER_EXPERIMENTS:
        for omega in PARAMETER_OMEGA_GRID:
            if omega == 0:
                continue

            for group_name, group_index in groups:
                parameter_omegas = [0.0, 0.0, 0.0]
                parameter_omegas[group_index] = omega

                experiments.append(
                    {
                        "name": (
                            f"parameter_"
                            f"{group_name}_"
                            f"{omega:g}"
                        ),
                        "noise_type": "parameter",
                        "group": group_name,
                        "value": omega,
                        "activation_sigmas": (
                            0.0,
                            0.0,
                            0.0,
                        ),
                        "parameter_omegas": tuple(
                            parameter_omegas
                        ),
                    }
                )

    return experiments


def append_result(result):
    file_exists = RESULTS_CSV.exists()

    with RESULTS_CSV.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(result.keys()),
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(result)


def get_completed_experiments():
    if not RESULTS_CSV.exists():
        return set()

    with RESULTS_CSV.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        return {
            row["name"]
            for row in csv.DictReader(file)
        }


def main():
    set_seed(SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}", flush=True)

    criterion = nn.CrossEntropyLoss()

    train_loader, test_loader = create_loaders(
        train_batch_size=256,
        test_batch_size=512,
        seed=SEED,
    )

    experiments = build_experiments()
    completed = get_completed_experiments()

    print(
        f"Количество экспериментов: {len(experiments)}",
        flush=True,
    )

    baseline_fgsm = None
    baseline_pgd = None

    for experiment in experiments:
        name = experiment["name"]

        activation_sigmas = experiment[
            "activation_sigmas"
        ]

        parameter_omegas = experiment[
            "parameter_omegas"
        ]

        checkpoint_path = (
            CHECKPOINT_DIR / f"{name}.pt"
        )

        print("\n" + "=" * 80, flush=True)
        print(f"Experiment: {name}", flush=True)
        print(
            f"Activation: {activation_sigmas}",
            flush=True,
        )
        print(
            f"Parameters: {parameter_omegas}",
            flush=True,
        )

        model = Net(
            activation_sigmas[0],
            activation_sigmas[1],
            activation_sigmas[2],
        ).to(device)

        if (
            name in completed
            and checkpoint_path.exists()
        ):
            print(
                "Загрузка сохранённой модели",
                flush=True,
            )

            checkpoint = torch.load(
                checkpoint_path,
                map_location=device,
                weights_only=False,
            )

            model.load_state_dict(
                checkpoint["model_state_dict"]
            )

            training_time = checkpoint[
                "training_time"
            ]

        else:
            optimizer = optim.SGD(
                model.parameters(),
                lr=LR,
                momentum=MOMENTUM,
            )

            start_time = time.time()

            train(
                net=model,
                train_loader=train_loader,
                test_loader=test_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                num_epochs=NUM_EPOCHS,
                omega1=parameter_omegas[0],
                omega2=parameter_omegas[1],
                omega3=parameter_omegas[2],
            )

            training_time = (
                time.time() - start_time
            )

            torch.save(
                {
                    "name": name,
                    "experiment": experiment,
                    "model_state_dict": (
                        model.state_dict()
                    ),
                    "training_time": training_time,
                },
                checkpoint_path,
            )

        clean_accuracy, clean_loss = evaluate(
            model=model,
            data_loader=test_loader,
            device=device,
            criterion=criterion,
        )

        if name == "baseline":
            baseline_fgsm = (
                create_adversarial_dataset(
                    attack_name="fgsm",
                    model=model,
                    data_loader=test_loader,
                    criterion=criterion,
                    device=device,
                    epsilon=(8 / 255) / 0.5,
                )
            )

            baseline_pgd = (
                create_adversarial_dataset(
                    attack_name="pgd",
                    model=model,
                    data_loader=test_loader,
                    criterion=criterion,
                    device=device,
                    epsilon=(8 / 255) / 0.5,
                    alpha=(2 / 255) / 0.5,
                    num_iter=10,
                )
            )

        if baseline_fgsm is None:
            raise RuntimeError(
                "FGSM-атака baseline не создана"
            )

        if baseline_pgd is None:
            raise RuntimeError(
                "PGD-атака baseline не создана"
            )

        fgsm_accuracy, _ = evaluate(
            model=model,
            data_loader=baseline_fgsm,
            device=device,
        )

        pgd_accuracy, _ = evaluate(
            model=model,
            data_loader=baseline_pgd,
            device=device,
        )

        result = {
            "name": name,
            "noise_type": experiment["noise_type"],
            "group": experiment["group"],
            "value": experiment["value"],

            "activation_sigma1": (
                activation_sigmas[0]
            ),
            "activation_sigma2": (
                activation_sigmas[1]
            ),
            "activation_sigma3": (
                activation_sigmas[2]
            ),

            "parameter_omega1": (
                parameter_omegas[0]
            ),
            "parameter_omega2": (
                parameter_omegas[1]
            ),
            "parameter_omega3": (
                parameter_omegas[2]
            ),

            "clean_loss": clean_loss,
            "clean_accuracy": clean_accuracy,
            "transfer_fgsm_accuracy": (
                fgsm_accuracy
            ),
            "transfer_pgd_accuracy": (
                pgd_accuracy
            ),
            "training_time_seconds": (
                training_time
            ),
        }

        if name not in completed:
            append_result(result)
            completed.add(name)

        print(result, flush=True)

        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nВсе эксперименты завершены")
    print(f"Результаты: {RESULTS_CSV.resolve()}")


if __name__ == "__main__":
    main()
