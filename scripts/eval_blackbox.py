import torch
import torch.nn as nn
import rootutils
import torchattacks
import torch.nn.functional as F
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


def load_model(ckpt, noise_sigma, device, arch="tile", tap="l1", noise_tap=None, dual_bn=False):
    kw = {"tap": tap, "noise_tap": noise_tap, "dual_bn": dual_bn} if arch == "tile" else {}
    m = _ARCHS[arch](num_classes=10, noise_sigma=noise_sigma, **kw).to(device)
    m.load_state_dict(torch.load(ckpt, map_location=device))
    m.eval()
    return Norm01(m, device).to(device).eval()


def get_loader01(n=None, batch_size=128, num_workers=4, seed=0):
    stl = load_dataset("jxie/stl10")["test"]
    ds = STL10RGBDataset(stl, transform=transforms.ToTensor())
    if n is not None:
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(len(ds), generator=g)[:n].tolist()
        ds = Subset(ds, idx)
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


def craft_transfer(surrogate01, loader, eps, device, steps=20):
    atk = torchattacks.PGD(surrogate01, eps=eps, alpha=eps * 2.5 / steps,
                           steps=steps, random_start=True)
    batches = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        batches.append((atk(x, y).detach(), y))
    return batches


@torch.no_grad()
def acc_on(model01, batches, reps=1):
    """Точность модели на заранее скрафченных батчах"""
    c = t = 0
    for x_adv, y in batches:
        logits = sum(model01(x_adv) for _ in range(reps)) / reps
        c += (logits.argmax(1) == y).sum().item()
        t += y.size(0)
    return c / t


def eot_pgd_acc(model01, loader, eps, device, steps=20, eot=10, reps=1):
    """
    White-box EOT-PGD атака:
      eot - число реализаций шума для усреднения .
      reps  — усреднение предсказания на eval.
    """
    model01.requires_grad_(False)      # атаке нужен градиент только по входу, будет быстрее backward
    alpha = eps * 2.5 / steps
    c = t = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        x0 = x.clone()
        x_adv = (x0 + torch.empty_like(x0).uniform_(-eps, eps)).clamp(0, 1)
        for _ in range(steps):
            g = torch.zeros_like(x_adv)
            for _ in range(max(1, eot)):
                xv = x_adv.detach().requires_grad_(True)
                with torch.enable_grad():
                    loss = F.cross_entropy(model01(xv), y)
                g = g + torch.autograd.grad(loss, xv)[0]
            g /= max(1, eot)
            x_adv = x_adv.detach() + alpha * g.sign()
            x_adv = torch.min(torch.max(x_adv, x0 - eps), x0 + eps).clamp(0, 1)
        with torch.no_grad():
            logits = sum(model01(x_adv) for _ in range(reps)) / reps
            c += (logits.argmax(1) == y).sum().item()
            t += y.size(0)
    return c / t
