import csv
from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from src.stimulus import add_noise

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
MAX_PIXEL = 255.0


def to_tensor(a):
    t = torch.from_numpy(a).float().unsqueeze(0) / MAX_PIXEL   # [1,h,half]
    t = t.repeat(3, 1, 1)                                      # [3,h,half]
    return (t - IMAGENET_MEAN) / IMAGENET_STD                  # как у STL


class STL10RGBDataset(Dataset):
    def __init__(self, hf_dataset, transform=None) -> None:
        super().__init__()
        self.dataset = hf_dataset
        self.transform = transform or transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = self.transform(item["image"].convert("RGB"))
        label = torch.tensor(item["label"], dtype=torch.long)
        return image, label


class SimkinDataset(Dataset):
    """Стимулы 8.1: кроп окна с обоими полями + запас -> ((left, right), log2 psi)."""

    def __init__(self, data_dir: Path, csv_path: Path, stimulus_cfg, crop_margin: int = 20):
        data_dir = Path(data_dir)
        csv_path = Path(csv_path)

        self.samples = []   # (path, psi)
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                path = data_dir / row["filename"]
                if not path.exists():
                    continue
                psi = float(row["psi"]) if row.get("psi") not in (None, "") else float("nan")
                self.samples.append((path, psi))

        # Фиксированное окно, охватывающее оба поля + запас crop_margin.
        s = stimulus_cfg
        center_x = s.canvas_width // 2
        x_left = center_x - s.margin - s.outer_box_size   # начало левого поля
        x_right = center_x + s.margin                     # начало правого поля
        self.x0 = max(x_left - crop_margin, 0)
        self.x1 = min(x_right + s.outer_box_size + crop_margin, s.canvas_width)
        self.y0 = max(s.center_y - crop_margin, 0)
        self.y1 = min(s.center_y + s.outer_box_size + crop_margin, s.canvas_height)
        self.stimulus_cfg = stimulus_cfg

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, psi = self.samples[idx]
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        img = img[self.y0:self.y1, self.x0:self.x1]   # окно с обоими полями
        img = add_noise(img, sigma=self.stimulus_cfg.sigma)

        W = img.shape[1]
        half = W // 2
        left, right = img[:, :half], img[:, W - half:]
        psi = torch.tensor(psi, dtype=torch.float32)
        return (to_tensor(left), to_tensor(right)), torch.log2(psi)