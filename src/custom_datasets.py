import csv
from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class STL10RGBDataset(Dataset):
    def __init__(self, hf_dataset) -> None:
        super().__init__()
        self.dataset = hf_dataset
        self.to_tensor = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"].convert("RGB")
        label = item["label"]
        image = self.to_tensor(image)
        label = torch.tensor(label, dtype=torch.long)
        return image, label


class SimkinDataset(Dataset):
    """Датасет стимулов 8.1 — возвращает (image_tensor, visible_label).

    image_tensor : float32 [1, H, W], нормализован в [0, 1]
    visible_label: float32 scalar (0.0 или 1.0) для BCEWithLogitsLoss
    """

    def __init__(self, data_dir: Path, csv_path: Path):
        data_dir = Path(data_dir)
        csv_path = Path(csv_path)

        labels: dict[str, int] = {}
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                labels[row["filename"]] = int(row["visible"])

        # Берём только те PNG, для которых есть метка в CSV
        self.samples = [
            (data_dir / fname, vis)
            for fname, vis in labels.items()
            if (data_dir / fname).exists()
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, visible = self.samples[idx]
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        img = torch.from_numpy(img).float().unsqueeze(0) / 255.0  # [1,H,W]
        img = img.repeat(3, 1, 1)                                 # [3,H,W]
        label = torch.tensor(visible, dtype=torch.float32)
        return img, label
        

        