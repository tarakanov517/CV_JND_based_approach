from itertools import cycle
from pathlib import Path

import torch
import rootutils

from omegaconf import OmegaConf
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm
from datasets import load_dataset

rootutils.setup_root(__file__, indicator="src", pythonpath=True)


from src.model import CustomResNet
from src.custom_datasets import SimkinDataset, STL10RGBDataset


train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(96, padding=8),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def train_one_epoch(model, stl_loader, simkin_iter, optimizer,
                    cls_criterion, simkin_criterion, lambda_simkin, device):
    model.train()

    total_cls_loss = 0.0
    total_simkin_loss = 0.0
    correct = 0
    total = 0

    for stl_imgs, stl_labels in tqdm(stl_loader, desc="train", leave=False):
        simkin_imgs, simkin_labels = next(simkin_iter)

        stl_imgs = stl_imgs.to(device)
        stl_labels = stl_labels.to(device)
        simkin_imgs = simkin_imgs.to(device)
        simkin_labels = simkin_labels.to(device)

        optimizer.zero_grad()

        # STL: классификация
        cls_logits = model(stl_imgs, mode='clas')
        loss_cls = cls_criterion(cls_logits, stl_labels)

        # Симкин: бинарная видимость
        simkin_logits = model(simkin_imgs, mode='simk').squeeze(1)
        loss_simkin = simkin_criterion(simkin_logits, simkin_labels)

        loss = loss_cls + lambda_simkin * loss_simkin
        loss.backward()
        optimizer.step()

        total_cls_loss += loss_cls.item() * stl_imgs.size(0)
        total_simkin_loss += loss_simkin.item() * simkin_imgs.size(0)
        preds = cls_logits.argmax(dim=1)
        correct += (preds == stl_labels).sum().item()
        total += stl_labels.size(0)

    return total_cls_loss / total, total_simkin_loss / total, correct / total


@torch.no_grad()
def evaluate(model, stl_loader, cls_criterion, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for imgs, labels in tqdm(stl_loader, desc="eval", leave=False):
        imgs = imgs.to(device)
        labels = labels.to(device)

        logits = model(imgs, mode='clas')
        loss = cls_criterion(logits, labels)

        total_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def train_model(
    train_stl_loader,
    test_stl_loader,
    train_simkin_loader,
    model_name,
    epochs=10,
    lr=1e-3,
    weight_decay=1e-4,
    lambda_simkin=1.0,
    device='cpu'
):
    model = CustomResNet().to(device)

    cls_criterion = nn.CrossEntropyLoss()
    simkin_criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    simkin_iter = cycle(train_simkin_loader)

    best_acc = 0.0
    history = {"train_loss": [], "train_simkin_loss": [], "train_acc": [],
               "test_loss": [], "test_acc": []}

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")

        train_loss, train_simkin_loss, train_acc = train_one_epoch(
            model, train_stl_loader, simkin_iter,
            optimizer, cls_criterion, simkin_criterion, lambda_simkin, device
        )
        test_loss, test_acc = evaluate(model, test_stl_loader, cls_criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_simkin_loss"].append(train_simkin_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        print(f"train cls_loss: {train_loss:.4f} | simkin_loss: {train_simkin_loss:.4f} | train acc: {train_acc:.4f}")
        print(f"test  loss: {test_loss:.4f} | test  acc: {test_acc:.4f}")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), f"{model_name}_best.pth")
            print(f"saved: {model_name}_best.pth | best acc: {best_acc:.4f}")

    return model, history


def main(cfg):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # STL-10
    dataset_stl  = load_dataset("jxie/stl10")
    train_rgb_stl = STL10RGBDataset(dataset_stl["train"], transform=train_transform)
    test_rgb_stl  = STL10RGBDataset(dataset_stl["test"],  transform=test_transform)

    simkin_full = SimkinDataset(
        data_dir=Path(cfg.simkin_data_dir),
        csv_path=Path(cfg.simkin_csv_path),
    )

    n_train = int(len(simkin_full) * cfg.train_size)
    n_test = len(simkin_full) - n_train

    train_simkin, test_simkin = random_split(
        simkin_full, [n_train, n_test],
        generator=torch.Generator().manual_seed(42)
    )

    pin = torch.cuda.is_available()

    train_stl_loader = DataLoader(
        train_rgb_stl, batch_size=cfg.batch_size_class,
        shuffle=True, drop_last=True,
        num_workers=cfg.num_workers, pin_memory=pin
    )
    test_stl_loader = DataLoader(
        test_rgb_stl, batch_size=cfg.batch_size_class,
        shuffle=False, drop_last=False,
        num_workers=cfg.num_workers, pin_memory=pin
    )
    train_simkin_loader = DataLoader(
        train_simkin, batch_size=cfg.batch_size_penalty,
        shuffle=True, drop_last=True,
        num_workers=cfg.num_workers, pin_memory=pin
    )

    train_model(
        train_stl_loader, test_stl_loader, train_simkin_loader,
        model_name=cfg.model_name,
        epochs=cfg.epochs,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        lambda_simkin=cfg.lambda_simkin,
        device=device,
    )


if __name__ == "__main__":
    cfg = OmegaConf.load("params.yaml")
    main(cfg.train)
