# python src/train_resnet.py        # параметры train из params.yaml
from pathlib import Path

import numpy as np
import torch
import rootutils
from scipy.stats import spearmanr
from omegaconf import OmegaConf
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm
from datasets import load_dataset

rootutils.setup_root(__file__, indicator="src", pythonpath=True)

from src.model import CustomResNet
from src.common import make_dir
from src.custom_datasets import SimkinDataset, STL10RGBDataset

THRES = -0.4

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


def _endless(loader):                 # бесконечный поток: пересоздаёт итератор -> свежий шум
    while True:
        for batch in loader:
            yield batch


def train_one_epoch(model, stl_loader, simkin_loader, optimizer,
                    cls_criterion, simkin_criterion, lambda_simkin, delta,
                    frontend_convs, initial_weights, device):
    model.train()
    total_cls_loss = total_simkin_loss = 0.0
    correct = total = simkin_seen = 0

    n_steps = max(len(stl_loader), len(simkin_loader))
    stl_it, simkin_it = _endless(stl_loader), _endless(simkin_loader)

    for _ in tqdm(range(n_steps), desc="train", leave=False):
        stl_imgs, stl_labels = next(stl_it)
        simkin_imgs, simkin_labels = next(simkin_it)
        stl_imgs, stl_labels = stl_imgs.to(device), stl_labels.to(device)
        simkin_imgs = tuple(t.to(device) for t in simkin_imgs)
        simkin_labels = simkin_labels.to(device)

        optimizer.zero_grad()
        cls_logits = model(stl_imgs, mode='clas')                   # STL: классификация
        loss_cls = cls_criterion(cls_logits, stl_labels)
        simkin_logits = model(simkin_imgs, mode='simk').squeeze(1)  # Симкин: регрессия log2(ψ)
        loss_simkin = simkin_criterion(simkin_logits, simkin_labels)
        loss = loss_cls + lambda_simkin * loss_simkin
        loss.backward()
        optimizer.step()

        if delta > 0:                                               # soft-freeze фронт-энда
            with torch.no_grad():
                for name, mod in frontend_convs:
                    w0 = initial_weights[name]
                    mod.weight.data.clamp_(w0 - delta, w0 + delta)

        total_cls_loss += loss_cls.item() * stl_imgs.size(0)
        total_simkin_loss += loss_simkin.item() * simkin_imgs[0].size(0)
        simkin_seen += simkin_imgs[0].size(0)
        correct += (cls_logits.argmax(dim=1) == stl_labels).sum().item()
        total += stl_labels.size(0)

    return total_cls_loss / total, total_simkin_loss / simkin_seen, correct / total


@torch.no_grad()
def evaluate(model, stl_loader, cls_criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in tqdm(stl_loader, desc="eval", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs, mode='clas')
        total_loss += cls_criterion(logits, labels).item() * imgs.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate_simkin(model, simkin_loader, simkin_criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_pred, all_tgt = [], []
    for imgs, labels in tqdm(simkin_loader, desc="eval-simkin", leave=False):
        imgs = tuple(t.to(device) for t in imgs)
        labels = labels.to(device)
        logits = model(imgs, mode='simk').squeeze(1)
        total_loss += simkin_criterion(logits, labels).item() * labels.size(0)
        correct += ((logits > THRES) == (labels > 0)).cpu().sum().item()
        total += labels.size(0)
        all_pred.append(logits.cpu())
        all_tgt.append(labels.cpu())

    preds = torch.cat(all_pred).numpy()     # корреляция pred vs истинный log2(ψ) — главная метрика головы
    tgts = torch.cat(all_tgt).numpy()
    m = ~np.isnan(tgts)
    preds, tgts = preds[m], tgts[m]
    pearson = float(np.corrcoef(preds, tgts)[0, 1])
    spearman = float(spearmanr(preds, tgts).correlation)
    print(f"  simkin corr: Pearson {pearson:.3f} | Spearman {spearman:.3f}")
    return total_loss / total, correct / total


def pretrain_simkin(model, simkin_loader, optimizer, criterion, device, epochs):
    """Low-LR прогрев на синтетике: только simkin-ветка (stem + layer1 + simkin_head)."""
    model.train()
    for ep in range(1, epochs + 1):
        total, seen = 0.0, 0
        for imgs, labels in tqdm(simkin_loader, desc=f"pretrain {ep}/{epochs}", leave=False):
            imgs = tuple(t.to(device) for t in imgs)
            labels = labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs, mode='simk').squeeze(1), labels)
            loss.backward()
            optimizer.step()
            total += loss.item() * labels.size(0)
            seen += labels.size(0)
        print(f"pretrain {ep}/{epochs} | simkin_loss: {total / seen:.4f}")


def train_model(train_stl_loader, test_stl_loader, train_simkin_loader, test_simkin_loader,
                model_name, epochs=10, lr=1e-3, weight_decay=1e-4, lambda_simkin=1.0,
                pretrain_epochs=0, pretrain_lr=1e-4, freeze_frontend=False, delta=0.01,
                noise_sigma=0.0, device='cpu'):
    model = CustomResNet(noise_sigma=noise_sigma).to(device)
    make_dir("models", delete_if_exist=False)
    cls_criterion = nn.CrossEntropyLoss()
    simkin_criterion = nn.MSELoss()

    # Фаза прогрева на Симкине: stem + layer1 + simkin_head
    if pretrain_epochs > 0:
        pre_params = (list(model.stem.parameters())
                      + list(model.layer1.parameters())
                      + list(model.simkin_head.parameters()))
        pre_opt = optim.AdamW(pre_params, lr=pretrain_lr, weight_decay=0.0)
        pretrain_simkin(model, train_simkin_loader, pre_opt, simkin_criterion, device, pretrain_epochs)
        print("=== после прогрева: simkin на тесте ===")
        evaluate_simkin(model, test_simkin_loader, simkin_criterion, device)

    if freeze_frontend:
        delta = 0.0

    # Снимок начальных весов фронт-энд-слоёв (stem + layer1) под soft-freeze
    frontend_convs, initial_weights = [], {}
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Conv2d) and any(name.startswith(p) for p in ["stem", "layer1"]):
            frontend_convs.append((name, mod))
            initial_weights[name] = mod.weight.data.clone()

    frontend_param_ids = {id(mod.weight) for _, mod in frontend_convs}
    optimizer = optim.AdamW([
        {"params": [mod.weight for _, mod in frontend_convs], "lr": 1e-5, "weight_decay": 0.0},
        {"params": [p for p in model.parameters() if id(p) not in frontend_param_ids],
         "lr": lr, "weight_decay": weight_decay},
    ])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    history = {"train_loss": [], "train_simkin_loss": [], "train_acc": [],
               "test_loss": [], "test_acc": []}

    for epoch in range(epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        train_loss, train_simkin_loss, train_acc = train_one_epoch(
            model, train_stl_loader, train_simkin_loader, optimizer,
            cls_criterion, simkin_criterion, lambda_simkin, delta,
            frontend_convs, initial_weights, device)
        test_loss, test_acc = evaluate(model, test_stl_loader, cls_criterion, device)
        simkin_test_loss, simkin_test_acc = evaluate_simkin(
            model, test_simkin_loader, simkin_criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_simkin_loss"].append(train_simkin_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        print(f"train cls_loss: {train_loss:.4f} | simkin_loss: {train_simkin_loss:.4f} | train acc: {train_acc:.4f}")
        print(f"test  loss: {test_loss:.4f} | test  acc: {test_acc:.4f}")
        print(f"simkin test loss: {simkin_test_loss:.4f} | simkin test acc: {simkin_test_acc:.4f}")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), f"models/{model_name}_best.pth")
            print(f"saved: {model_name}_best.pth | best acc: {best_acc:.4f}")

    return model, history


def main(cfg):
    tcfg = cfg.train
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset_stl = load_dataset("jxie/stl10")
    train_rgb_stl = STL10RGBDataset(dataset_stl["train"], transform=train_transform)
    test_rgb_stl = STL10RGBDataset(dataset_stl["test"], transform=test_transform)

    simkin_full = SimkinDataset(
        data_dir=Path(tcfg.simkin_data_dir),
        csv_path=Path(tcfg.simkin_csv_path),
        stimulus_cfg=cfg.stimulus,
    )
    n_train = int(len(simkin_full) * tcfg.train_size)
    train_simkin, test_simkin = random_split(
        simkin_full, [n_train, len(simkin_full) - n_train],
        generator=torch.Generator().manual_seed(42))

    pin = torch.cuda.is_available()
    train_stl_loader = DataLoader(train_rgb_stl, batch_size=tcfg.batch_size_class, shuffle=True,
                                  drop_last=True, num_workers=tcfg.num_workers, pin_memory=pin)
    test_stl_loader = DataLoader(test_rgb_stl, batch_size=tcfg.batch_size_class, shuffle=False,
                                 drop_last=False, num_workers=tcfg.num_workers, pin_memory=pin)
    train_simkin_loader = DataLoader(train_simkin, batch_size=tcfg.batch_size_penalty, shuffle=True,
                                     drop_last=True, num_workers=tcfg.num_workers, pin_memory=pin)
    test_simkin_loader = DataLoader(test_simkin, batch_size=tcfg.batch_size_penalty, shuffle=False,
                                    drop_last=False, num_workers=0, pin_memory=pin)

    train_model(
        train_stl_loader, test_stl_loader, train_simkin_loader, test_simkin_loader,
        model_name=tcfg.model_name, epochs=tcfg.epochs, lr=tcfg.lr,
        weight_decay=tcfg.weight_decay, lambda_simkin=tcfg.lambda_simkin,
        pretrain_epochs=tcfg.pretrain_epochs, pretrain_lr=tcfg.pretrain_lr,
        freeze_frontend=tcfg.freeze_frontend, noise_sigma=tcfg.frontend_noise,
        delta=tcfg.delta, device=device)


if __name__ == "__main__":
    main(OmegaConf.load("params.yaml"))