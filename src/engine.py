import csv
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torchattacks
from torchvision import transforms

from data import create_loaders
from eval import eval
from model import Net
from train import DendriteNoise, train


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def create_model(
    sigma_lateral,
    sigma_prop,
    sigma_add,
    pyramidal_sigma,
    pyramidal_gamma,
    pyramidal_b,
    device,
):
    return Net(
        sigma_lateral=sigma_lateral,
        sigma_prop=sigma_prop,
        sigma_add=sigma_add,
        sigma=pyramidal_sigma,
        gamma=pyramidal_gamma,
        b=pyramidal_b,
    ).to(device)


def fit(
    model,
    optimizer,
    criterion,
    train_loader,
    test_loader,
    device,
    epochs,
    model_path,
    train_parameters,
):
    best_loss = float("inf")
    dendrite_noise = DendriteNoise()

    for epoch in range(epochs):
        train_loss, train_accuracy = train(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_loader=train_loader,
            device=device,
            dendrite_noise=dendrite_noise,
            **train_parameters,
        )
        test_loss, test_accuracy = eval(
            model=model,
            test_loader=test_loader,
            criterion=criterion,
            device=device,
        )

        print(
            f"{epoch + 1}/{epochs} "
            f"train_loss={train_loss:.6f} "
            f"train_accuracy={train_accuracy:.6f} "
            f"test_loss={test_loss:.6f} "
            f"test_accuracy={test_accuracy:.6f}"
        )

        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model, model_path)


def get_accuracy(model, loader, criterion, device):
    _, accuracy = eval(
        model=model,
        test_loader=loader,
        criterion=criterion,
        device=device,
    )
    return accuracy


def engine(
    dataset_name,
    lr,
    momentum,
    device,
    num_epochs,
    train_batch_size,
    test_batch_size,
    sigma_lateral,
    sigma_prop,
    sigma_add,
    pyramidal_sigma,
    pyramidal_gamma,
    pyramidal_b,
    omega1,
    omega2,
    omega3,
    dendrite_theta1,
    dendrite_sigma1,
    dendrite_theta2,
    dendrite_sigma2,
    dendrite_theta3,
    dendrite_sigma3,
    seed=42,
    experiment_name="human_noise",
):
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    os.makedirs("experiments", exist_ok=True)

    set_seed(seed)

    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    train_loader, test_loader = create_loaders(
        dataset_name,
        train_batch_size,
        test_batch_size,
        train_transform,
        test_transform,
    )

    criterion = nn.CrossEntropyLoss()
    baseline_path = "models/model_baseline.tar"

    if not os.path.exists(baseline_path):
        baseline = create_model(
            sigma_lateral=0.0,
            sigma_prop=0.0,
            sigma_add=0.0,
            pyramidal_sigma=0.0,
            pyramidal_gamma=pyramidal_gamma,
            pyramidal_b=pyramidal_b,
            device=device,
        )
        baseline_optimizer = torch.optim.SGD(
            baseline.parameters(),
            lr=lr,
            momentum=momentum,
        )
        fit(
            model=baseline,
            optimizer=baseline_optimizer,
            criterion=criterion,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            epochs=num_epochs,
            model_path=baseline_path,
            train_parameters={},
        )

    baseline = torch.load(
        baseline_path,
        map_location=device,
        weights_only=False,
    )
    baseline.to(device)
    baseline.eval()

    fgsm_path = "data/adv_data_fgsm.pt"
    attack_fgsm = torchattacks.FGSM(baseline, eps=8 / 255)
    attack_fgsm.set_normalization_used(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    )
    if not os.path.exists(fgsm_path):
        attack_fgsm.save(
            data_loader=test_loader,
            save_path=fgsm_path,
            verbose=True,
        )
    fgsm_loader = attack_fgsm.load(load_path=fgsm_path)

    pgd_path = "data/adv_data_pgd.pt"
    attack_pgd = torchattacks.PGD(
        baseline,
        eps=8 / 255,
        alpha=2 / 255,
        steps=10,
        random_start=True,
    )
    attack_pgd.set_normalization_used(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    )
    if not os.path.exists(pgd_path):
        attack_pgd.save(
            data_loader=test_loader,
            save_path=pgd_path,
            verbose=True,
        )
    pgd_loader = attack_pgd.load(load_path=pgd_path)

    results_path = "experiments/results.csv"

    if not os.path.exists(results_path):
        baseline_accuracy = get_accuracy(
            baseline,
            test_loader,
            criterion,
            device,
        )
        baseline_fgsm = get_accuracy(
            baseline,
            fgsm_loader,
            criterion,
            device,
        )
        baseline_pgd = get_accuracy(
            baseline,
            pgd_loader,
            criterion,
            device,
        )

        with open(results_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(
                (
                    "Model",
                    "Seed",
                    "Sigma Lateral",
                    "Sigma Prop",
                    "Sigma Add",
                    "Pyramidal Sigma",
                    "Pyramidal Gamma",
                    "Pyramidal B",
                    "Omega1",
                    "Omega2",
                    "Omega3",
                    "Dendrite Theta1",
                    "Dendrite Sigma1",
                    "Dendrite Theta2",
                    "Dendrite Sigma2",
                    "Dendrite Theta3",
                    "Dendrite Sigma3",
                    "Accuracy",
                    "Attacked Accuracy FGSM",
                    "Attacked Accuracy PGD",
                )
            )
            writer.writerow(
                (
                    "baseline",
                    seed,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    pyramidal_gamma,
                    pyramidal_b,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    baseline_accuracy,
                    baseline_fgsm,
                    baseline_pgd,
                )
            )

    with open(results_path, newline="", encoding="utf-8") as file:
        completed = {row["Model"] for row in csv.DictReader(file)}

    if experiment_name in completed:
        print(f"Skipping completed experiment: {experiment_name}")
        return

    set_seed(seed)
    model = create_model(
        sigma_lateral=sigma_lateral,
        sigma_prop=sigma_prop,
        sigma_add=sigma_add,
        pyramidal_sigma=pyramidal_sigma,
        pyramidal_gamma=pyramidal_gamma,
        pyramidal_b=pyramidal_b,
        device=device,
    )
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
    )
    model_path = f"models/model_{experiment_name}.tar"

    train_parameters = {
        "omega1": omega1,
        "omega2": omega2,
        "omega3": omega3,
        "dendrite_theta1": dendrite_theta1,
        "dendrite_sigma1": dendrite_sigma1,
        "dendrite_theta2": dendrite_theta2,
        "dendrite_sigma2": dendrite_sigma2,
        "dendrite_theta3": dendrite_theta3,
        "dendrite_sigma3": dendrite_sigma3,
    }
    fit(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        epochs=num_epochs,
        model_path=model_path,
        train_parameters=train_parameters,
    )

    model = torch.load(
        model_path,
        map_location=device,
        weights_only=False,
    )
    model.to(device)

    accuracy = get_accuracy(model, test_loader, criterion, device)
    accuracy_fgsm = get_accuracy(model, fgsm_loader, criterion, device)
    accuracy_pgd = get_accuracy(model, pgd_loader, criterion, device)

    with open(results_path, "a", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow(
            (
                experiment_name,
                seed,
                sigma_lateral,
                sigma_prop,
                sigma_add,
                pyramidal_sigma,
                pyramidal_gamma,
                pyramidal_b,
                omega1,
                omega2,
                omega3,
                dendrite_theta1,
                dendrite_sigma1,
                dendrite_theta2,
                dendrite_sigma2,
                dendrite_theta3,
                dendrite_sigma3,
                accuracy,
                accuracy_fgsm,
                accuracy_pgd,
            )
        )

    print(
        f"{experiment_name}: "
        f"clean={accuracy:.6f} "
        f"fgsm={accuracy_fgsm:.6f} "
        f"pgd={accuracy_pgd:.6f}"
    )
