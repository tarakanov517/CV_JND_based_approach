import argparse
from engine import engine
import torch

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="Программа для запуска обучения с Гауссовым шумом и атаками на модель",
    )
    parser.add_argument(
        "-e", "--epochs", action="store", help="Количество эпох обучения", type=int
    )
    parser.add_argument(
        "-lr", "--learning_rate", action="store", help="Learning rate", type=float
    )

    parser.add_argument("-m", "--momentum", action="store", help="Momentum", type=float)

    parser.add_argument(
        "-train_bs",
        "--train_batch_size",
        action="store",
        help="Размер батча для train",
        type=int,
    )

    parser.add_argument(
        "-test_bs",
        "--test_batch_size",
        action="store",
        help="Размер батча для test",
        type=int,
    )

    parser.add_argument(
        "-dn",
        "--dataset_name",
        default="uoft-cs/cifar10",
        action="store",
        help="Имя датасета",
    )

    parser.add_argument(
        "-s1", "--sigma1", action="store", help="Sigma для первого блока", type=float
    )

    parser.add_argument(
        "-s2", "--sigma2", action="store", help="Sigma для второго блока", type=float
    )

    parser.add_argument(
        "-s3", "--sigma3", action="store", help="Sigma для третьего блока", type=float
    )

    parser.add_argument(
        "-o1", "--omega1", action="store", help="Omega для первого блока", type=float
    )

    parser.add_argument(
        "-o2", "--omega2", action="store", help="Omega для второго блока", type=float
    )

    parser.add_argument(
        "-o3", "--omega3", action="store", help="Omega для третьего блока", type=float
    )

    parser.add_argument(
        "--noise_schedule",
        choices=("fixed", "linear"),
        default="fixed",
    )

    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--ramp_epochs",
        type=int,
        default=10,
    )

    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    print("device:", device)

    args = parser.parse_args()
    engine(
        dataset_name=args.dataset_name,
        lr=args.learning_rate,
        momentum=args.momentum,
        device=device,
        num_epochs=args.epochs,
        train_batch_size=args.train_batch_size,
        test_batch_size=args.test_batch_size,
        sigma1=args.sigma1,
        sigma2=args.sigma2,
        sigma3=args.sigma3,
        omega1=args.omega1,
        omega2=args.omega2,
        omega3=args.omega3,
        noise_schedule=args.noise_schedule,
        warmup_epochs=args.warmup_epochs,
        ramp_epochs=args.ramp_epochs,
    )
