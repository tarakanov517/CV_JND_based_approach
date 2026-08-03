import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import rootutils
from omegaconf import OmegaConf
from torch import optim
from torch.utils.data import DataLoader, random_split
from datasets import load_dataset
from tqdm import tqdm

rootutils.setup_root(__file__, indicator="src", pythonpath=True)

from src.custom_datasets import STL10RGBDataset
from src.tile_dataset import SingleTileDataset
from src.tile_model import TileResNet
from src.train_tile import pretrain_simkin, train_tf, test_tf, eval_clean, eval_robust

TAILS = {
    'l4': {'l4'},
    'l3_l4': {'l3', 'l4'},
    'l2_l3_l4': {'l2', 'l3', 'l4'},
}


def _endless(loader):
    while True:
        for batch in loader:
            yield batch


def set_trainable(model, tail):
    """Замораживает всё, кроме стадий из tail; головы всегда обучаемы."""
    for n, m in model._stages:
        req = n in tail
        for p in m.parameters():
            p.requires_grad_(req)
    for p in model.class_head.parameters():
        p.requires_grad_(True)
    for p in model.simkin_head.parameters():
        p.requires_grad_(True)


def train_classifier(model, tr_loader, te_loader, epochs, lr, device,
                     multitask=False, simk_loader=None, lam=0.4):
    best = 0.0
    criterion = nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = optim.AdamW(params, lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    simk_it = _endless(simk_loader) if multitask else None
    for ep in range(1, epochs + 1):
        model.train()
        for x, y in tqdm(tr_loader, desc=f"E{ep}/{epochs}", leave=False):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = criterion(model(x, mode='clas'), y)
            if multitask:
                xs, ys = next(simk_it)
                xs, ys = xs.to(device), ys.to(device)
                loss = loss + lam * F.mse_loss(model(xs, mode='simk').squeeze(1), ys)
            loss.backward()
            opt.step()
        sched.step()
        acc = eval_clean(model, te_loader, device)
        best = max(best, acc)
        print(f"  Epoch {ep:02d} | test_acc={acc:.4f}")
    return model


def run(tile_dir, tile_csv, sigma, pretrain_epochs=30, pretrain_lr=1e-3,
        cls_epochs=10, ft_epochs=10, lr=1e-3, lam=0.4, noise_sigma=0.5,
        batch_size=128, num_workers=8, eps=2 / 255, pgd_steps=20, eot_k=20,
        tails=('l4', 'l3_l4', 'l2_l3_l4'), seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tile_full = SingleTileDataset(tile_dir, tile_csv, sigma=sigma, fixed_seed=None)
    n_val = max(1, int(0.2 * len(tile_full)))
    tile_tr, _ = random_split(tile_full, [len(tile_full) - n_val, n_val],
                              generator=torch.Generator().manual_seed(seed))
    simk_loader = DataLoader(tile_tr, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers, drop_last=True)
    stl = load_dataset("jxie/stl10")
    tr_loader = DataLoader(STL10RGBDataset(stl["train"], transform=train_tf),
                           batch_size=batch_size, shuffle=True,
                           num_workers=num_workers, drop_last=True)
    te_loader = DataLoader(STL10RGBDataset(stl["test"], transform=test_tf),
                           batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # 1-я фаза: обучаем Симкин-голов на l4
    print("\n[Фаза 1] Симкин-претрейн, голова на l4")
    model_A = TileResNet(num_classes=10, noise_sigma=0.0, tap='l4', noise_tap='conv1').to(device)
    pretrain_simkin(model_A, simk_loader, pretrain_epochs, pretrain_lr, device)
    simkin_head_w = {k: v.detach().clone() for k, v in model_A.simkin_head.state_dict().items()}
    del model_A
    torch.cuda.empty_cache()

    # 2-я фаза: обучаем базовый классификатор с нуля (Симкин и шум off)
    print("\n[Фаза 2] Базовый классификатор с нуля")
    model_B = TileResNet(num_classes=10, noise_sigma=0.0, tap='l4', noise_tap='conv1').to(device)
    set_trainable(model_B, {n for n, _ in model_B._stages})   # всё обучаемо
    train_classifier(model_B, tr_loader, te_loader, cls_epochs, lr, device)
    base_state = {k: v.detach().clone() for k, v in model_B.state_dict().items()}
    del model_B
    torch.cuda.empty_cache()

    # 3-я фаза: мультитаск-дообучение хвоста (обычный классификатор + Симкин), шум на conv1
    results = {}
    for tail_name in tails:
        print(f"\n[Фаза 3] tail={tail_name}, шум на conv1 (σ={noise_sigma})")
        
        model_C = TileResNet(num_classes=10, noise_sigma=noise_sigma,
                             tap='l4', noise_tap='conv1').to(device)
        model_C.load_state_dict(base_state)                       # веса базового классификатора
        model_C.simkin_head.load_state_dict(simkin_head_w)        # Симкин-голова из фазы 1
        
        set_trainable(model_C, TAILS[tail_name])                  # мороз фронт-энда
        
        train_classifier(model_C, tr_loader, te_loader, ft_epochs, lr, device,
                         multitask=True, simk_loader=simk_loader, lam=lam)

        clean = eval_clean(model_C, te_loader, device)
        r_eot = eval_robust(model_C, te_loader, eps, pgd_steps, device, k_eot=eot_k)
        results[tail_name] = (clean, r_eot)
        torch.save(model_C.state_dict(), f"models/ResNetLateSimkin_{tail_name}_best.pt")
        print(f"  [{tail_name}] clean={clean:.4f} | EOT-PGD(k={eot_k})={r_eot:.4f}")
        del model_C
        torch.cuda.empty_cache()

    print("\n── Итог (шум на conv1) ──")
    for t, (c, r) in results.items():
        print(f"{t:10s} clean={c:.4f}  robust={r:.4f}")
    return results


def main():
    cfg = OmegaConf.load("params.yaml")
    p = cfg.train_tile
    run(tile_dir=p.tile_dir, tile_csv=p.tile_csv, sigma=cfg.noise.sigma,
        pretrain_epochs=p.pretrain_epochs, pretrain_lr=p.pretrain_lr,
        cls_epochs=p.epochs, ft_epochs=p.epochs, lr=p.lr, noise_sigma=p.noise_sigma,
        batch_size=p.batch_size, num_workers=p.num_workers,
        eps=p.eps_255 / 255, pgd_steps=p.pgd_steps, eot_k=p.eot_k, seed=p.seed)


if __name__ == "__main__":
    main()
