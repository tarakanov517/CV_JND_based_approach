import torch
import torch.nn as nn
import rootutils
import torchattacks
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from datasets import load_dataset

rootutils.setup_root(__file__, indicator="src", pythonpath=True)

from src.custom_datasets import STL10RGBDataset
from src.tile_model import TileResNet
from src.model import CustomResNet

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class Norm01(nn.Module):
    """Оборачивает модель так, чтобы принимать [0,1]."""

    def __init__(self, model, device):
        super().__init__()
        self.model = model
        self.register_buffer("mean", MEAN.to(device))
        self.register_buffer("std", STD.to(device))

    def forward(self, x):
        return self.model((x - self.mean) / self.std)


_ARCHS = {"tile": TileResNet, "siamese": CustomResNet}


def load_model(ckpt, noise_sigma, device, arch="tile"):
    m = _ARCHS[arch](num_classes=10, noise_sigma=noise_sigma).to(device)
    m.load_state_dict(torch.load(ckpt, map_location=device))
    m.eval()
    return Norm01(m, device).to(device).eval()


def get_loader01(n=1000, batch_size=128, num_workers=4):
    stl = load_dataset("jxie/stl10")["test"]
    ds = STL10RGBDataset(stl, transform=transforms.ToTensor())
    if n is not None:
        ds = Subset(ds, range(n))
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)


@torch.no_grad()
def _clean_acc(model01, loader, device):
    c = t = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        c += (model01(x).argmax(1) == y).sum().item()
        t += y.size(0)
    return c / t


def square_acc(model01, loader, eps, device, n_queries=2000, n_restarts=1):
    atk = torchattacks.Square(model01, norm="Linf", eps=eps,
                              n_queries=n_queries, n_restarts=n_restarts)
    c = t = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        x_adv = atk(x, y)
        with torch.no_grad():
            c += (model01(x_adv).argmax(1) == y).sum().item()
        t += y.size(0)
    return c / t


def transfer_acc(target01, surrogate01, loader, eps, device, steps=20, reps=1):
    atk = torchattacks.PGD(surrogate01, eps=eps, alpha=eps * 2.5 / steps,
                           steps=steps, random_start=True)
    c = t = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        x_adv = atk(x, y)
        with torch.no_grad():
            logits = sum(target01(x_adv) for _ in range(reps)) / reps
            c += (logits.argmax(1) == y).sum().item()
        t += y.size(0)
    return c / t
