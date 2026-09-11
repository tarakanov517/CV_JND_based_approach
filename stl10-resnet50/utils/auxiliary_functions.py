import torch
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import transforms, models
from datasets import load_dataset
from pathlib import Path
import numpy as np
import torchvision.transforms.functional as TF
import random
from datasets import Dataset as HFDataset
import torch.nn as nn
import os
from tqdm.auto import tqdm

class STL10Dataset(Dataset):

    def __init__(self, hf_dataset: HFDataset, matrix_path: Path | None = None, is_jnd: bool = False, layers: str | None = None, is_train: bool = True):
        self.hf_dataset = hf_dataset
        self.matrix_path = matrix_path
        self.is_jnd = is_jnd
        self.layers = layers
        self.is_train = is_train

        if is_jnd:
            self.jnd_matrices = np.load(matrix_path, mmap_mode='r')
        else:
            self.jnd_matrices = None

    def __len__(self):
        return len(self.hf_dataset)

    def transform(self, image: torch.Tensor) -> torch.Tensor:
        if not self.is_train:
            return image

        if torch.rand(1).item() > 0.5:
            image = TF.hflip(image)

        angle = random.uniform(-10.0, 10.0)
        image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.NEAREST)

        image = TF.pad(image, padding=4, padding_mode="reflect") 

        _, h, w = image.shape
        crop_h, crop_w = 96, 96

        top = random.randint(0, h - crop_h)
        left = random.randint(0, w - crop_w)

        image = TF.crop(image, top, left, crop_h, crop_w)

        return image

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        label = item['label']

        if self.is_jnd:
            matrix = np.array(self.jnd_matrices[idx]) # (x, y, z, k)
            if self.layers == 'kkk':
                matrix = np.repeat(matrix[..., 3:4], repeats=3, axis=-1)
            elif self.layers == 'xzk':
                matrix = matrix[..., [0, 2, 3]]
            else:
                raise ValueError('Неизвестный режим слоев')
            image = torch.from_numpy(matrix).permute(2, 0, 1).float()
        else:
            image = TF.to_tensor(item["image"].convert("RGB"))

        image = self.transform(image)
        label = torch.tensor(label, dtype=torch.long)

        return image, label

def get_loader01(n=None, batch_size=128, num_workers=4, seed=0):
    stl = load_dataset("jxie/stl10")["test"]
    ds = STL10Dataset(stl, is_jnd=False, is_train=False)
    if n is not None:
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(len(ds), generator=g)[:n].tolist()
        ds = Subset(ds, idx)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

def make_backbone():
    m = models.resnet50(weights="IMAGENET1K_V1")
    m.fc = nn.Linear(m.fc.in_features, 10)
    return m

def train_eval(model, optimizer, criterion, scheduler, epochs, train_loader, val_loader, model_name, device):
    loses_history = []
    metric_history = []
    best_acc = 0.0
    scaler = torch.amp.GradScaler('cuda')

    BASE_DIR = Path('/home/misavinov/scratch/ws/my_space/stl10-resnet50')
    weights_path = BASE_DIR / 'weights'

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0

        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]'):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / len(train_loader)
        loses_history.append(avg_train_loss)

        model.eval()
        correct_predictions = 0
        total_samples = 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]'):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, dim=1)
                correct_predictions += torch.sum(preds == labels).item()
                total_samples += labels.size(0)

        epoch_accuracy = correct_predictions / total_samples
        metric_history.append(epoch_accuracy)
        scheduler.step()

        if epoch_accuracy > best_acc:
            best_acc = epoch_accuracy
            torch.save(model.state_dict(), os.path.join(weights_path, f'{model_name}.pth'))

        print(f'Epoch: {epoch + 1}/{epochs}. Train loss: {avg_train_loss:.4f}. Accuracy: {epoch_accuracy:.4f}')

    model.load_state_dict(torch.load(os.path.join(weights_path, f'{model_name}.pth')))
    return loses_history, metric_history