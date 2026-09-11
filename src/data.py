import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class CaltechDataset(Dataset):
    def __init__(self, split, transform):
        self.split = split
        self.transform = transform

    def __len__(self):
        return len(self.split)

    def __getitem__(self, index):
        item = self.split[index]
        image = item["image"].convert("RGB")
        return self.transform(image), int(item["label"]) - 1


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)


def create_loaders(batch_size, workers, seed, dataset_name="ilee0022/Caltech-256"):
    dataset = load_dataset(dataset_name)
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        CaltechDataset(dataset["train"], train_transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    validation_loader = DataLoader(
        CaltechDataset(dataset["validation"], test_transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        CaltechDataset(dataset["test"], test_transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )
    return train_loader, validation_loader, test_loader
