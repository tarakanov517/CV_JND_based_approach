import argparse

import torch

from engine import engine


def build_parser():
    parser = argparse.ArgumentParser(
        description="Обучение модели с человекоподобными шумами и проверкой на атаках",
    )

    parser.add_argument("-e", "--epochs", type=int, default=40)
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.01)
    parser.add_argument("-m", "--momentum", type=float, default=0.9)
    parser.add_argument("-train_bs", "--train_batch_size", type=int, default=32)
    parser.add_argument("-test_bs", "--test_batch_size", type=int, default=32)
    parser.add_argument("-dn", "--dataset_name", default="uoft-cs/cifar10")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiment_name", default="human_noise")

    parser.add_argument("--sigma_lateral", type=float, default=0.0)
    parser.add_argument("--sigma_prop", type=float, default=0.0)
    parser.add_argument("--sigma_add", type=float, default=0.0)

    parser.add_argument("--pyramidal_sigma", type=float, default=0.0)
    parser.add_argument("--pyramidal_gamma", type=float, default=1.0)
    parser.add_argument("--pyramidal_b", type=float, default=1.0)

    parser.add_argument("--omega1", type=float, default=0.0)
    parser.add_argument("--omega2", type=float, default=0.0)
    parser.add_argument("--omega3", type=float, default=0.0)

    parser.add_argument("--dendrite_theta1", type=float, default=0.0)
    parser.add_argument("--dendrite_sigma1", type=float, default=0.0)
    parser.add_argument("--dendrite_theta2", type=float, default=0.0)
    parser.add_argument("--dendrite_sigma2", type=float, default=0.0)
    parser.add_argument("--dendrite_theta3", type=float, default=0.0)
    parser.add_argument("--dendrite_sigma3", type=float, default=0.0)

    return parser


def main():
    args = build_parser().parse_args()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    engine(
        dataset_name=args.dataset_name,
        lr=args.learning_rate,
        momentum=args.momentum,
        device=device,
        num_epochs=args.epochs,
        train_batch_size=args.train_batch_size,
        test_batch_size=args.test_batch_size,
        sigma_lateral=args.sigma_lateral,
        sigma_prop=args.sigma_prop,
        sigma_add=args.sigma_add,
        pyramidal_sigma=args.pyramidal_sigma,
        pyramidal_gamma=args.pyramidal_gamma,
        pyramidal_b=args.pyramidal_b,
        omega1=args.omega1,
        omega2=args.omega2,
        omega3=args.omega3,
        dendrite_theta1=args.dendrite_theta1,
        dendrite_sigma1=args.dendrite_sigma1,
        dendrite_theta2=args.dendrite_theta2,
        dendrite_sigma2=args.dendrite_sigma2,
        dendrite_theta3=args.dendrite_theta3,
        dendrite_sigma3=args.dendrite_sigma3,
        seed=args.seed,
        experiment_name=args.experiment_name,
    )


if __name__ == "__main__":
    main()
