import random

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class CifarDataset(Dataset):
    def __init__(self, data, transforms):
        self.data = data
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        image = item["img"]
        label = item["label"]

        if self.transforms:
            image = self.transforms(image)
        return image, label


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32

    random.seed(worker_seed)
    np.random.seed(worker_seed)


def create_loaders(
    train_batch_size=32,
    test_batch_size=32,
    dataset_name="uoft-cs/cifar10",
    seed=42,
):
    transform_train = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    transform_test = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    dataset = load_dataset(dataset_name)

    cifar10_train = CifarDataset(dataset["train"], transforms=transform_train)

    cifar10_test = CifarDataset(dataset["test"], transforms=transform_test)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        cifar10_train,
        batch_size=train_batch_size,
        num_workers=8,
        pin_memory=True,
        shuffle=True,
        generator=g,
        worker_init_fn=seed_worker,
    )

    test_loader = DataLoader(
        cifar10_test,
        num_workers=8,
        pin_memory=True,
        batch_size=test_batch_size,
        shuffle=False,
        worker_init_fn=seed_worker,
    )

    return train_loader, test_loader
