"""Классификационные датасеты для всех экспериментов.

    stl10       -- 96x96, HF jxie/stl10 (старый пайплайн, трансформы 1:1 как в train_tile)
    imagenette  -- нативные 224, fast.ai imagenette2-320 (10 простых классов ImageNet)
    imagewoof   -- нативные 224, fast.ai imagewoof2-320 (10 пород собак, сложнее, нет потолка ~99%)

Train-лоадеры отдают ImageNet-нормализованные тензоры (как раньше),
eval-лоадеры для атак (get_eval_loader01) -- тензоры в [0, 1] (нормализацию делает обёртка Norm01).
"""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from torchvision.datasets.utils import download_and_extract_archive

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

DATASETS = {
    "stl10": dict(img_size=96, num_classes=10),
    "imagenette": dict(img_size=224, num_classes=10,
                       url="https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz",
                       folder="imagenette2-320"),
    "imagewoof": dict(img_size=224, num_classes=10,
                      url="https://s3.amazonaws.com/fast-ai-imageclas/imagewoof2-320.tgz",
                      folder="imagewoof2-320"),
}


def img_size(name: str) -> int:
    return DATASETS[name]["img_size"]


def make_transforms(name: str, train: bool, normalize: bool = True):
    norm = [transforms.Normalize(MEAN, STD)] if normalize else []
    if name == "stl10":                              # ровно как в src/train_tile.py
        if train:
            return transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(96, padding=8),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
                transforms.ToTensor(), *norm])
        return transforms.Compose([transforms.ToTensor(), *norm])

    s = DATASETS[name]["img_size"]                   # 224: стандартный ImageNet-рецепт
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(s, scale=(0.35, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.ToTensor(), *norm])
    return transforms.Compose([
        transforms.Resize(int(round(s * 256 / 224))),
        transforms.CenterCrop(s),
        transforms.ToTensor(), *norm])


def _fastai_root(name: str, root: str | Path, download: bool = True) -> Path:
    meta = DATASETS[name]
    root = Path(root)
    path = root / meta["folder"]
    if not path.exists():
        if not download:
            raise FileNotFoundError(path)
        download_and_extract_archive(meta["url"], download_root=str(root))
    return path


def get_dataset(name: str, split: str, transform, root: str | Path = "data"):
    """split: 'train' | 'test'  (для imagenette/imagewoof 'test' == их 'val')."""
    if name == "stl10":
        from datasets import load_dataset
        from src.custom_datasets import STL10RGBDataset
        stl = load_dataset("jxie/stl10")
        return STL10RGBDataset(stl["train" if split == "train" else "test"], transform=transform)
    if name in ("imagenette", "imagewoof"):
        path = _fastai_root(name, root)
        return datasets.ImageFolder(path / ("train" if split == "train" else "val"), transform=transform)
    raise ValueError(f"unknown dataset {name}")


def get_loaders(name: str, batch_size: int, num_workers: int, root: str | Path = "data"):
    """(train_loader, test_loader) с ImageNet-нормализацией -- для обучения."""
    pin = torch.cuda.is_available()
    tr = get_dataset(name, "train", make_transforms(name, train=True), root)
    te = get_dataset(name, "test", make_transforms(name, train=False), root)
    tr_loader = DataLoader(tr, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                           drop_last=True, pin_memory=pin, persistent_workers=num_workers > 0)
    te_loader = DataLoader(te, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                           pin_memory=pin)
    return tr_loader, te_loader


def random_subset(ds, n: int | None, seed: int = 0):
    """Случайная (не первые N!) подвыборка -- тест отсортирован по классам."""
    if n is None or n >= len(ds):
        return ds
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(ds), generator=g)[:n].tolist()
    return Subset(ds, idx)


def get_eval_loader01(name: str, n: int | None = None, seed: int = 0, batch_size: int = 128,
                      num_workers: int = 4, root: str | Path = "data"):
    """Тест в [0,1] без нормализации -- вход для Norm01-обёрток и атак."""
    ds = get_dataset(name, "test", make_transforms(name, train=False, normalize=False), root)
    return DataLoader(random_subset(ds, n, seed), batch_size=batch_size, shuffle=False,
                      num_workers=num_workers)


class UInt8Dataset(Dataset):
    """Тест, закешированный как uint8 NHWC (нужен для corruptions: imagecorruptions работает с HWC uint8)."""

    def __init__(self, images: np.ndarray, labels: np.ndarray):
        self.images, self.labels = images, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        x = torch.from_numpy(self.images[i]).permute(2, 0, 1).float() / 255.0
        return x, int(self.labels[i])


def load_test_uint8(name: str, root: str | Path = "data", cache_dir: str | Path = "cache"):
    """Весь тест как (N,H,W,3) uint8 + метки; кешируется в cache_dir/{name}_test_uint8.npz."""
    cache = Path(cache_dir) / f"{name}_test_uint8.npz"
    if cache.exists():
        d = np.load(cache)
        return d["images"], d["labels"]
    ds = get_dataset(name, "test", make_transforms(name, train=False, normalize=False), root)
    imgs, labels = [], []
    for x, y in DataLoader(ds, batch_size=256, num_workers=4):
        imgs.append((x.permute(0, 2, 3, 1).numpy() * 255.0).round().astype(np.uint8))
        labels.append(np.asarray(y))
    images, labels = np.concatenate(imgs), np.concatenate(labels)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, images=images, labels=labels)
    return images, labels
