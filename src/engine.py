from data import create_loaders
from model import Net
from train import train
from eval import eval
from torchvision import transforms
import contextlib
import torch.nn as nn
import torch
import numpy as np
import torchattacks
import os
import csv
import random


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def get_noise_multiplier(epoch, noise_schedule, warmup_epochs, ramp_epochs):
    if noise_schedule == "fixed":
        return 1.0
    else:
        if epoch < warmup_epochs:
            return 0.0
        progress = (epoch - warmup_epochs + 1) / ramp_epochs

        return min(progress, 1.0)


def engine(
    dataset_name,
    lr,
    momentum,
    device,
    num_epochs,
    train_batch_size,
    test_batch_size,
    sigma1,
    sigma2,
    sigma3,
    omega1,
    omega2,
    omega3,
    noise_schedule="fixed",
    warmup_epochs=5,
    ramp_epochs=10,
    seed=42,
):

    set_seed(seed)
    transforms_train = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    transforms_test = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    train_loader, test_loader = create_loaders(
        dataset_name,
        train_batch_size,
        test_batch_size,
        transforms_train,
        transforms_test,
    )

    criterion = nn.CrossEntropyLoss()

    if not os.path.exists("./models/model_baseline.tar"):
        model_baseline = Net(0, 0, 0)
        optimizer = torch.optim.SGD(
            model_baseline.parameters(), lr=lr, momentum=momentum
        )
        model_baseline.train()
        model_baseline.to(device)

        eval_max_loss = float("inf")

        for epoch in range(num_epochs):
            train_loss, train_accuracy = train(
                model=model_baseline,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                train_loader=train_loader,
                omega1=0,
                omega2=0,
                omega3=0,
            )

            val_loss, val_accuracy = eval(
                model=model_baseline,
                test_loader=test_loader,
                criterion=criterion,
                device=device,
            )
            print(
                f"{epoch+1}, train loss={train_loss}, train accuracy={train_accuracy}"
            )
            print(f"test loss={val_loss}, test accuracy={val_accuracy}")

            if val_loss < eval_max_loss:
                eval_max_loss = val_loss
                torch.save(model_baseline, "./models/model_baseline.tar")

    model_baseline = torch.load(
        "./models/model_baseline.tar",
        map_location=device,
        weights_only=False,
    )
    model_baseline.to(device)

    loss, accuracy = eval(
        model=model_baseline,
        test_loader=test_loader,
        criterion=criterion,
        device=device,
    )

    normalization_mean = (0.5, 0.5, 0.5)
    normalization_std = (0.5, 0.5, 0.5)

    model_baseline.eval()

    attack_fgsm = torchattacks.FGSM(model_baseline, eps=8 / 255)

    attack_fgsm.set_normalization_used(mean=normalization_mean, std=normalization_std)

    if not os.path.exists("./data/adv_data_fgsm.pt"):

        attack_fgsm.save(
            data_loader=test_loader, save_path="./data/adv_data_fgsm.pt", verbose=True
        )

    with contextlib.redirect_stdout(open(os.devnull, "w")):
        adv_loader_fgsm = attack_fgsm.load(load_path="./data/adv_data_fgsm.pt")

    attacked_loss_fgsm, attacked_accuracy_fgsm = eval(
        model=model_baseline,
        test_loader=adv_loader_fgsm,
        criterion=criterion,
        device=device,
    )

    print(
        f"Baseline - loss fgsm: {attacked_loss_fgsm}, accuracy fgsm: {attacked_accuracy_fgsm}"
    )

    attack_pgd = torchattacks.PGD(
        model_baseline,
        eps=8 / 255,
        alpha=2 / 255,
        steps=10,
        random_start=True,
    )

    attack_pgd.set_normalization_used(mean=normalization_mean, std=normalization_std)

    if not os.path.exists("./data/adv_data_pgd.pt"):

        attack_pgd.save(
            data_loader=test_loader, save_path="./data/adv_data_pgd.pt", verbose=True
        )

    with contextlib.redirect_stdout(open(os.devnull, "w")):
        adv_loader_pgd = attack_pgd.load(load_path="./data/adv_data_pgd.pt")

    attacked_loss_pgd, attacked_accuracy_pgd = eval(
        model=model_baseline,
        test_loader=adv_loader_pgd,
        criterion=criterion,
        device=device,
    )

    print(
        f"Baseline - loss pgd: {attacked_loss_pgd}, accuracy pgd: {attacked_accuracy_pgd}"
    )
    if not os.path.exists("./experiments/results.csv"):
        with open("./experiments/results.csv", "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                (
                    "Model",
                    "Accuracy",
                    "Attacked Accuracy FGSM",
                    "Attacked Accuracy PGD",
                )
            )
            writer.writerow(
                (
                    "Baseline Model",
                    accuracy,
                    attacked_accuracy_fgsm,
                    attacked_accuracy_pgd,
                )
            )

    set_seed(seed)
    
    
    model = Net(sigma1=sigma1, sigma2=sigma2, sigma3=sigma3).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    if noise_schedule == "fixed":
        experiment_name = (
            f"Model_{sigma1}_{sigma2}_{sigma3}_" f"{omega1}_{omega2}_{omega3}"
        )
    else:
        experiment_name = (
            f"Model_{num_epochs}_{noise_schedule}_"
            f"w{warmup_epochs}_r{ramp_epochs}_"
            f"{sigma1}_{sigma2}_{sigma3}_"
            f"{omega1}_{omega2}_{omega3}"
        )
    model_path = f"./models/{experiment_name}.tar"

    eval_max_loss = float("inf")

    for epoch in range(num_epochs):

        noise_multiplier = get_noise_multiplier(
            epoch=epoch,
            noise_schedule=noise_schedule,
            warmup_epochs=warmup_epochs,
            ramp_epochs=ramp_epochs,
        )

        current_sigma1 = sigma1 * noise_multiplier
        current_sigma2 = sigma2 * noise_multiplier
        current_sigma3 = sigma3 * noise_multiplier

        current_omega1 = omega1 * noise_multiplier
        current_omega2 = omega2 * noise_multiplier
        current_omega3 = omega3 * noise_multiplier

        model.set_activation_sigmas(
            current_sigma1,
            current_sigma2,
            current_sigma3,
        )

        print(
            f"epoch={epoch + 1}, "
            f"noise_multiplier={noise_multiplier:.4f}, "
            f"sigma=({current_sigma1:.6g}, "
            f"{current_sigma2:.6g}, "
            f"{current_sigma3:.6g}), "
            f"omega=({current_omega1:.6g}, "
            f"{current_omega2:.6g}, "
            f"{current_omega3:.6g})"
        )
        train_loss, train_accuracy = train(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            train_loader=train_loader,
            omega1=current_omega1,
            omega2=current_omega2,
            omega3=current_omega3,
        )

        val_loss, val_accuracy = eval(
            model=model,
            test_loader=test_loader,
            criterion=criterion,
            device=device,
        )
        print(f"{epoch+1}, train loss={train_loss}, train accuracy={train_accuracy}")
        print(f"test loss={val_loss}, test accuracy={val_accuracy}")

        full_noise_reached = noise_multiplier >= 1.0

        checkpoint_allowed = noise_schedule == "fixed" or full_noise_reached

        if checkpoint_allowed and val_loss < eval_max_loss:
            eval_max_loss = val_loss
            torch.save(model, model_path)

    model_path = model_path

    model = torch.load(
        model_path,
        map_location=device,
        weights_only=False,
    )
    model.to(device)
    attacked_loss_fgsm, attacked_accuracy_fgsm = eval(
        model=model,
        test_loader=adv_loader_fgsm,
        criterion=criterion,
        device=device,
    )

    print(
        f"Loss noise model {attacked_loss_fgsm}, accuracy noise model {attacked_accuracy_fgsm}"
    )

    attacked_loss_pgd, attacked_accuracy_pgd = eval(
        model=model,
        test_loader=adv_loader_pgd,
        criterion=criterion,
        device=device,
    )

    print(
        f"Loss noise model {attacked_loss_pgd}, accuracy noise model {attacked_accuracy_pgd}"
    )

    loss_model, accuracy_model = eval(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=device,
    )
    with open("./experiments/results.csv", "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            (
                experiment_name,
                accuracy_model,
                attacked_accuracy_fgsm,
                attacked_accuracy_pgd,
            )
        )
