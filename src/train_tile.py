import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import rootutils
from omegaconf import OmegaConf
from torch import optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm
from datasets import load_dataset

rootutils.setup_root(__file__, indicator="src", pythonpath=True)

from src.custom_datasets import STL10RGBDataset
from src.tile_dataset import SingleTileDataset
from src.tile_model import TileResNet

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

train_tf = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(96, padding=8),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
test_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def pretrain_simkin(model, loader, epochs, lr, device):
    params = (list(model.stem.parameters()) + list(model.layer1.parameters())
              + list(model.simkin_head.parameters()))
    opt = optim.AdamW(params, lr=lr, weight_decay=0.0)
    crit = nn.MSELoss()
    for ep in range(1, epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for x, y in tqdm(loader, desc=f"pretrain {ep}/{epochs}", leave=False):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            pred = model(x, mode='simk').squeeze(1)
            loss = crit(pred, y)
            loss.backward()
            opt.step()
            tot += loss.item() * x.size(0)
            n += x.size(0)
        print(f"[pretrain {ep}/{epochs}] MSE(log₂ψ) = {tot / n:.4f}")


def snapshot_frontend(model):
    return {n: p.detach().clone() for n, p in model.named_parameters()
            if (n.startswith("stem") or n.startswith("layer1")) and p.dim() > 1}


def clamp_frontend(model, w0, delta):
    if delta <= 0:
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in w0:
                    p.copy_(w0[n])
        return
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in w0:
                p.copy_(torch.max(torch.min(p, w0[n] + delta), w0[n] - delta))


def _bounds(device):
    mean, std = MEAN.to(device), STD.to(device)
    return mean, std, (0.0 - mean) / std, (1.0 - mean) / std


def pgd(model, x, y, eps, alpha, steps, device, k_eot=0):
    mean, std, lo, hi = _bounds(device)
    eps_n, alpha_n = eps / std, alpha / std
    x_adv = x.clone().detach() + torch.empty_like(x).uniform_(-1, 1) * eps_n
    x_adv = torch.clamp(x_adv, lo, hi)
    for _ in range(steps):
        grad = torch.zeros_like(x_adv)
        for _ in range(max(1, k_eot)):
            with torch.enable_grad():
                xv = x_adv.detach().requires_grad_(True)
                loss = F.cross_entropy(model(xv), y)
                loss.backward()
            grad += xv.grad.detach()
        grad /= max(1, k_eot)
        with torch.no_grad():
            x_adv = x_adv + alpha_n * grad.sign()
            x_adv = torch.min(torch.max(x_adv, x - eps_n), x + eps_n)
            x_adv = torch.clamp(x_adv, lo, hi)
    return x_adv.detach()


@torch.no_grad()
def eval_clean(model, loader, device):
    model.eval()
    c = t = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        c += (model(x).argmax(1) == y).sum().item()
        t += y.size(0)
    return c / t


def eval_robust(model, loader, eps, steps, device, k_eot=0, n_batches=None):
    model.eval()
    alpha = eps * 2.5 / steps
    c = t = 0
    for i, (x, y) in enumerate(tqdm(loader, desc="robust", leave=False)):
        if n_batches is not None and i >= n_batches:
            break
        x, y = x.to(device), y.to(device)
        x_adv = pgd(model, x, y, eps, alpha, steps, device, k_eot=k_eot)
        with torch.no_grad():
            c += (model(x_adv).argmax(1) == y).sum().item()
        t += y.size(0)
    return c / t


def run_training(
    tile_dir="output_solo_exp/dataset_tile",
    tile_csv="output_solo_exp/labels_tile.csv",
    no_pretrain=False,
    pretrain_epochs=30,
    pretrain_lr=1e-3,
    soft_delta=0.01,
    noise_sigma=0.0,
    epochs=10,
    lr=1e-3,
    batch_size=128,
    num_workers=8,
    eps=2 / 255,
    pgd_steps=20,
    eot_k=10,
    robust_batches=None,
    save_path=None,
    name=None,
    seed=42,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    cfg = OmegaConf.load("params.yaml")
    print(f"Device: {device} | name={name} noise={noise_sigma} δ={soft_delta} pretrain={not no_pretrain}")

    tile_full = SingleTileDataset(tile_dir, tile_csv, sigma=cfg.noise.sigma, fixed_seed=None)
    n_val = max(1, int(0.2 * len(tile_full)))
    tile_tr, _ = random_split(tile_full, [len(tile_full) - n_val, n_val],
                              generator=torch.Generator().manual_seed(seed))
    tile_loader = DataLoader(tile_tr, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers, drop_last=True)

    stl = load_dataset("jxie/stl10")
    tr_loader = DataLoader(STL10RGBDataset(stl["train"], transform=train_tf),
                           batch_size=batch_size, shuffle=True,
                           num_workers=num_workers, drop_last=True)
    te_loader = DataLoader(STL10RGBDataset(stl["test"], transform=test_tf),
                           batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = TileResNet(num_classes=10, noise_sigma=noise_sigma).to(device)
    if not no_pretrain:
        pretrain_simkin(model, tile_loader, pretrain_epochs, pretrain_lr, device)

    w0 = snapshot_frontend(model)
    print(f"[soft-freeze] snapshot {len(w0)} conv-тензоров stem+layer1, δ={soft_delta}")

    crit = nn.CrossEntropyLoss()
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    save_path = save_path or f"models/ResNetTile{'_noisy' if noise_sigma > 0 else ''}_best.pt"

    best = 0.0
    for ep in range(1, epochs + 1):
        model.train()
        tot, c, n = 0.0, 0, 0
        for x, y in tqdm(tr_loader, desc=f"E{ep}/{epochs}", leave=False):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x, mode='clas')
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            clamp_frontend(model, w0, soft_delta)
            tot += loss.item() * x.size(0)
            c += (logits.argmax(1) == y).sum().item()
            n += y.size(0)
        sched.step()
        acc = eval_clean(model, te_loader, device)
        print(f"Epoch {ep:02d} | train_loss={tot/n:.4f} train_acc={c/n:.4f} | test_acc={acc:.4f}")
        if acc > best:
            best = acc
            torch.save(model.state_dict(), save_path)

    # оценка модели
    model.load_state_dict(torch.load(save_path, map_location=device))
    clean = eval_clean(model, te_loader, device)
    r_pgd = eval_robust(model, te_loader, eps, pgd_steps, device, k_eot=0, n_batches=robust_batches)
    r_eot = None
    if noise_sigma > 0 and eot_k > 0:
        r_eot = eval_robust(model, te_loader, eps, pgd_steps, device,
                            k_eot=eot_k, n_batches=robust_batches)

    honest = r_eot if r_eot is not None else r_pgd
    attack = "EOT-PGD" if r_eot is not None else "PGD"
    res = {
        "name": name or f"tile_ns{noise_sigma}_d{soft_delta}{'_noPre' if no_pretrain else ''}",
        "clean": clean, "pgd": r_pgd, "eot": r_eot, "honest": honest,
        "attack": attack, "noise_sigma": noise_sigma, "soft_delta": soft_delta,
        "eps": eps,
    }
    print(f"\n── Итог [{res['name']}] (eps={eps*255:.0f}/255) ──")
    print(f"clean = {clean:.4f} | PGD = {r_pgd:.4f}"
          + (f" | EOT-PGD(k={eot_k}) = {r_eot:.4f}" if r_eot is not None else ""))
    return res


def main():
    cfg = OmegaConf.load("params.yaml")
    p = OmegaConf.to_container(cfg.train_tile, resolve=True)
    p["eps"] = p.pop("eps_255") / 255.0       # с дробными числами в yaml напряженка
    run_training(**p)


if __name__ == "__main__":
    main()
