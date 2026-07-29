import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.nn as nn


class ImageDataset(Dataset):
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


def create_loaders(
    dataset_name, train_batch_size, test_batch_size, transforms_train, transforms_test
):

    dataset = load_dataset(dataset_name)

    train_dataset = ImageDataset(dataset["train"], transforms=transforms_train)
    test_dataset = ImageDataset(dataset["test"], transforms=transforms_test)

    train_loader = DataLoader(
        train_dataset, batch_size=train_batch_size, num_workers=8, shuffle=True
    )

    test_loader = DataLoader(
        test_dataset, batch_size=test_batch_size, num_workers=8, shuffle=False
    )

    return train_loader, test_loader
