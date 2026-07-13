from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import rootutils
from torch.utils.data import Dataset

rootutils.setup_root(__file__, indicator="src", pythonpath=True)

from src.stimulus import add_noise

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class SingleTileDataset(Dataset):
    def __init__(self, data_dir, csv_path, sigma, fixed_seed=None):
        self.dir = Path(data_dir)
        df = pd.read_csv(csv_path)
        self.samples = [(r.filename, float(np.log2(r.psi))) for r in df.itertuples()]
        self.sigma = float(sigma)
        self.fixed_seed = fixed_seed

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        fname, y = self.samples[i]
        img = cv2.imread(str(self.dir / fname), cv2.IMREAD_GRAYSCALE) 
        if self.fixed_seed is not None:
            np.random.seed(self.fixed_seed + i)
            
        img = add_noise(img, self.sigma)
        
        t = img.astype(np.float32) / 255.0
        t = np.stack([t, t, t], axis=0)
        t = (t - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]
        return torch.from_numpy(t), torch.tensor(y, dtype=torch.float32)
