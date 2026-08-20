import argparse
import contextlib
import csv
import os
import statistics

import torch
import torchattacks
from torchvision import transforms

from data import create_loaders


MODELS = {
    "baseline": "model_baseline.tar",
    "activation_block1": "Model_0.15_0.0_0.0_0.0_0.0_0.0.tar",
    "parameters_block1": "Model_0.0_0.0_0.0_0.002_0.0_0.0.tar",
    "parameters_blocks": "Model_0.0_0.0_0.0_0.002_0.001_0.003.tar",
}

MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)


def loader_accuracy(model, loader, device):
    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            predictions = model(images).argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return correct / total


def adversarial_accuracy(model, loader, attack, device):
    correct = 0
    total = 0

    model.eval()
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        adversarial_images = attack(images, labels)

        with torch.no_grad():
            predictions = model(adversarial_images).argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return correct / total


def fgsm(model):
    attack = torchattacks.FGSM(model, eps=8 / 255)
    attack.set_normalization_used(mean=MEAN, std=STD)
    return attack


def pgd(model, steps):
    attack = torchattacks.PGD(
        model,
        eps=8 / 255,
        alpha=2 / 255,
        steps=steps,
        random_start=True,
    )
    attack.set_normalization_used(mean=MEAN, std=STD)
    return attack


def load_attack_data(model, path, attack_name):
    attack = fgsm(model) if attack_name == "fgsm" else pgd(model, 10)

    with open(os.devnull, "w") as null_output, contextlib.redirect_stdout(
        null_output
    ):
        return attack.load(load_path=path)


def set_attack_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="check_seeds")
    parser.add_argument("--dataset", default="uoft-cs/cifar10")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seeds", default="42,43,44,45,46")
    args = parser.parse_args()

    seeds = [int(value) for value in args.seeds.split(",")]
    root = os.path.abspath(args.root)
    output_dir = os.path.join(root, "evaluation")
    os.makedirs(output_dir, exist_ok=True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    _, test_loader = create_loaders(
        args.dataset,
        args.batch_size,
        args.batch_size,
        test_transform,
        test_transform,
    )

    raw_path = os.path.join(output_dir, "raw_results.csv")
    values = {}

    with open(raw_path, "w", newline="") as raw_file:
        writer = csv.writer(raw_file)
        writer.writerow(
            (
                "Model",
                "Model Seed",
                "Evaluation",
                "Attack",
                "Attack Source Seed",
                "Accuracy",
            )
        )

        def save_result(model_name, model_seed, evaluation, attack, source_seed, accuracy):
            writer.writerow(
                (model_name, model_seed, evaluation, attack, source_seed, accuracy)
            )
            raw_file.flush()
            metric = evaluation if attack == "clean" else f"{evaluation}_{attack}"
            values.setdefault((model_name, model_seed, metric), []).append(accuracy)

        for model_seed in seeds:
            for model_name, filename in MODELS.items():
                model_path = os.path.join(
                    root,
                    f"seed_{model_seed}",
                    "models",
                    filename,
                )
                model = torch.load(
                    model_path,
                    map_location=device,
                    weights_only=False,
                )
                model.to(device)
                model.eval()

                clean_accuracy = loader_accuracy(model, test_loader, device)
                save_result(
                    model_name,
                    model_seed,
                    "clean",
                    "clean",
                    "",
                    clean_accuracy,
                )

                whitebox_attacks = (
                    ("fgsm", fgsm(model)),
                    ("pgd10", pgd(model, 10)),
                    ("pgd50", pgd(model, 50)),
                )

                for attack_index, (attack_name, attack) in enumerate(whitebox_attacks):
                    set_attack_seed(100000 + model_seed * 10 + attack_index)
                    accuracy = adversarial_accuracy(
                        model,
                        test_loader,
                        attack,
                        device,
                    )
                    save_result(
                        model_name,
                        model_seed,
                        "whitebox",
                        attack_name,
                        model_seed,
                        accuracy,
                    )

                for source_seed in seeds:
                    if source_seed == model_seed:
                        continue

                    source_data = os.path.join(root, f"seed_{source_seed}", "data")

                    for attack_name, filename in (
                        ("fgsm", "adv_data_fgsm.pt"),
                        ("pgd10", "adv_data_pgd.pt"),
                    ):
                        attack_loader = load_attack_data(
                            model,
                            os.path.join(source_data, filename),
                            attack_name,
                        )
                        accuracy = loader_accuracy(model, attack_loader, device)
                        save_result(
                            model_name,
                            model_seed,
                            "transfer",
                            attack_name,
                            source_seed,
                            accuracy,
                        )

                print(f"Завершено: model={model_name}, seed={model_seed}")

    per_seed = {
        key: statistics.mean(metric_values) for key, metric_values in values.items()
    }
    metrics = (
        "clean",
        "transfer_fgsm",
        "transfer_pgd10",
        "whitebox_fgsm",
        "whitebox_pgd10",
        "whitebox_pgd50",
    )
    summary_path = os.path.join(output_dir, "summary.csv")

    with open(summary_path, "w", newline="") as summary_file:
        writer = csv.writer(summary_file)
        writer.writerow(
            (
                "Model",
                "Metric",
                "Mean",
                "Standard Deviation",
                "Delta vs Baseline",
                "Wins vs Baseline",
            )
        )

        for model_name in MODELS:
            for metric in metrics:
                model_values = [
                    per_seed[(model_name, seed, metric)] for seed in seeds
                ]
                baseline_values = [
                    per_seed[("baseline", seed, metric)] for seed in seeds
                ]
                mean_value = statistics.mean(model_values)
                standard_deviation = statistics.stdev(model_values)
                delta = statistics.mean(
                    model_value - baseline_value
                    for model_value, baseline_value in zip(
                        model_values,
                        baseline_values,
                    )
                )
                wins = sum(
                    model_value > baseline_value
                    for model_value, baseline_value in zip(
                        model_values,
                        baseline_values,
                    )
                )

                writer.writerow(
                    (
                        model_name,
                        metric,
                        mean_value,
                        standard_deviation,
                        delta,
                        "" if model_name == "baseline" else wins,
                    )
                )

    print(f"Сырые результаты: {raw_path}")
    print(f"Сводные результаты: {summary_path}")


if __name__ == "__main__":
    main()
