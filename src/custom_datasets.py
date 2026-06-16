import csv
from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class STL10RGBDataset(Dataset):
    def __init__(self, hf_dataset, transform=None) -> None:
        super().__init__()
        self.dataset = hf_dataset
        self.transform = transform or transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"].convert("RGB")
        label = item["label"]
        image = self.transform(image)
        label = torch.tensor(label, dtype=torch.long)
        return image, label


class SimkinDataset(Dataset):
    """Датасет стимулов 8.1 — возвращает (image_tensor, visible_label).
    
    Возвращает кроп фиксированного окна с обоими полями 128 x 128, с запасом,
    
    image_tensor : float32 [3, h, w] в [0, 1]
    visible_label: float32 scalar (0.0 или 1.0) для BCEWithLogitsLoss
    """

    def __init__(self, data_dir: Path, csv_path: Path, stimulus_cfg, crop_margin: int = 20):
        data_dir = Path(data_dir)
        csv_path = Path(csv_path)

        labels: dict[str, int] = {}
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                labels[row["filename"]] = int(row["visible"])

        self.samples = [
            (data_dir / fname, vis)
            for fname, vis in labels.items()
            if (data_dir / fname).exists()
        ]

        # Фиксированное окно, охватывающее оба поля + запас crop_margin.
        s = stimulus_cfg
        center_x = s.canvas_width // 2
        x_left = center_x - s.margin - s.outer_box_size   # начало левого поля
        x_right = center_x + s.margin                     # начало правого поля
        m = crop_margin
        self.x0 = max(x_left - m, 0)
        self.x1 = min(x_right + s.outer_box_size + m, s.canvas_width)
        self.y0 = max(s.center_y - m, 0)
        self.y1 = min(s.center_y + s.outer_box_size + m, s.canvas_height)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, visible = self.samples[idx]
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        img = img[self.y0:self.y1, self.x0:self.x1]               # кроп окна с обоими полями
        img = torch.from_numpy(img).float().unsqueeze(0) / 255.0  # [1,h,w]
        img = img.repeat(3, 1, 1)                                 # [3,h,w]
        label = torch.tensor(visible, dtype=torch.float32)
        return img, label
    