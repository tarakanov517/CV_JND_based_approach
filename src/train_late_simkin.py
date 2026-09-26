"""Late-Simkin: голова Симкина на l4, нейронный шум на conv1.

Фазы:
  1. Симкин-претрейн (голова на l4)                     -> кеш {cache}/{ds}_simkin_head_l4_s{seed}.pt
  2. Базовый классификатор (Симкин off, шум off)       -> кеш {cache}/{ds}_base_s{seed}.pt
     Это же и есть BASELINE: ImageNet-pretrain ResNet50, дообученный тем же рецептом.
  3. Мультитаск-дообучение хвоста + шум на conv1       -> {out}/{ds}_LateSimkin_{tail}_ns{σ}_lam{λ}_s{seed}.pt

Фазы 1–2 от σ не зависят, поэтому кешируются и переиспользуются (noise-ablation гоняет только фазу 3).

Запуск (параметры = секция late_simkin из params.yaml + пресет + CLI-оверрайды OmegaConf):
    python src/train_late_simkin.py                                   # STL-10 96, как раньше
    python src/train_late_simkin.py preset=exp224                     # Imagenette 224
    python src/train_late_simkin.py preset=exp224 noise_sigmas=[0.5,1.0] tails=[l4]
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import rootutils
from omegaconf import OmegaConf
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

rootutils.setup_root(__file__, indicator="src", pythonpath=True)

from src.cls_data import get_loaders
from src.tile_dataset import SingleTileDataset
from src.tile_model import TileResNet
from src.train_tile import pretrain_simkin, eval_clean

torch.backends.cudnn.benchmark = True

TAILS = {
    'l4': {'l4'},
    'l3_l4': {'l3', 'l4'},
    'l2_l3_l4': {'l2', 'l3', 'l4'},
}
MODEL_KW = dict(tap='l4', noise_tap='conv1', dual_bn=True)    # для load_model в eval


def _endless(loader):                 # бесконечный поток: пересоздаёт итератор -> свежий shuffle/шум
    while True:
        for batch in loader:
            yield batch


def freeze_bn_stats(model):           # BN в eval -> running-stats не обновляются (иначе тайл-форварды их портят)
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()


def frozen_pairs(model, tail):
    """Пары (param, снимок) для conv-весов стадий вне tail — под soft-freeze."""
    pairs = []
    for n, m in model._stages:
        if n not in tail:
            for p in m.parameters():
                if p.dim() > 1:                 # только веса свёрток (BN не трогаем)
                    pairs.append((p, p.detach().clone()))
    return pairs


def clamp_frozen(pairs, delta):                 # вернуть conv-веса в коридор +- δ вокруг снимка
    with torch.no_grad():
        for p, w0 in pairs:
            p.clamp_(w0 - delta, w0 + delta)


def train_classifier(model, tr_loader, te_loader, epochs, lr, device,
                     multitask=False, simk_loader=None, lam=0.4,
                     frozen=None, delta=0.0, freeze_bn=False, amp=False):
    """Возвращает историю: список dict(epoch, train_loss, train_acc, test_acc)."""
    criterion = nn.CrossEntropyLoss()
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)   # все параметры обучаемы, мороз — клэмпом
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    simk_it = _endless(simk_loader) if multitask else None
    use_amp = amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    hist = []
    for ep in range(1, epochs + 1):
        model.train()
        if freeze_bn:                 # не портим BN-статистики
            freeze_bn_stats(model)
        tot, c, n = 0.0, 0, 0
        for x, y in tqdm(tr_loader, desc=f"E{ep}/{epochs}", leave=False):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(x, mode='clas')
                loss = criterion(logits.float(), y)
                if multitask:
                    xs, ys = next(simk_it)
                    xs, ys = xs.to(device, non_blocking=True), ys.to(device, non_blocking=True)
                    loss = loss + lam * F.mse_loss(model(xs, mode='simk').squeeze(1).float(), ys)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if frozen and delta > 0:            # soft-freeze: возврат conv-весов в +-δ
                clamp_frozen(frozen, delta)
            tot += loss.item() * y.size(0)
            c += (logits.argmax(1) == y).sum().item()
            n += y.size(0)
        sched.step()
        acc = eval_clean(model, te_loader, device)
        hist.append(dict(epoch=ep, train_loss=tot / n, train_acc=c / n, test_acc=acc))
        print(f"  Epoch {ep:02d} | loss={tot/n:.4f} train_acc={c/n:.4f} | test_acc={acc:.4f}")
    return hist


def build_loaders(cfg):
    """(simk_loader, tr_loader, te_loader)."""
    tile_full = SingleTileDataset(cfg.tile_dir, cfg.tile_csv, sigma=cfg.render_sigma, fixed_seed=None)
    n_val = max(1, int(0.2 * len(tile_full)))
    tile_tr, _ = random_split(tile_full, [len(tile_full) - n_val, n_val],
                              generator=torch.Generator().manual_seed(cfg.get("split_seed", 42)))  # сплит тайлов фиксирован
    simk_loader = DataLoader(tile_tr, batch_size=cfg.batch_size, shuffle=True,
                             num_workers=cfg.num_workers, drop_last=True,
                             pin_memory=torch.cuda.is_available())
    tr_loader, te_loader = get_loaders(cfg.dataset, cfg.batch_size, cfg.num_workers, cfg.data_root)
    return simk_loader, tr_loader, te_loader


def _cache_path(cfg, what, seed):
    return Path(cfg.cache_dir) / f"{cfg.dataset}_{what}_s{seed}.pt"


def new_model(noise_sigma=0.0, num_classes=10):
    return TileResNet(num_classes=num_classes, noise_sigma=noise_sigma, **MODEL_KW)


def get_phase1_head(cfg, simk_loader, device, seed):
    """Фаза 1: Симкин-претрейн, голова на l4. Возвращает state_dict головы (кешируется)."""
    path = _cache_path(cfg, "simkin_head_l4", seed)
    if path.exists():
        print(f"[Фаза 1] кеш: {path}")
        return torch.load(path, map_location="cpu")
    print("\n[Фаза 1] Симкин-претрейн, голова на l4")
    torch.manual_seed(seed)
    model_A = new_model().to(device)
    pretrain_simkin(model_A, simk_loader, cfg.pretrain_epochs, cfg.pretrain_lr, device, amp=cfg.amp)
    head = {k: v.detach().cpu().clone() for k, v in model_A.simkin_head.state_dict().items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(head, path)
    del model_A
    torch.cuda.empty_cache()
    return head


def get_phase2_base(cfg, tr_loader, te_loader, device, seed):
    """Фаза 2: базовый классификатор (== baseline). Возвращает state_dict (кешируется)."""
    path = _cache_path(cfg, "base", seed)
    if path.exists():
        print(f"[Фаза 2] кеш: {path}")
        return torch.load(path, map_location="cpu")
    print("\n[Фаза 2] Базовый классификатор (Симкин off, шум off) == baseline")
    torch.manual_seed(seed)
    model_B = new_model().to(device)
    train_classifier(model_B, tr_loader, te_loader, cfg.cls_epochs, cfg.lr, device, amp=cfg.amp)
    state = {k: v.detach().cpu().clone() for k, v in model_B.state_dict().items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    base_out = Path(cfg.out_dir) / f"{cfg.dataset}_baseline_s{seed}.pt"   # отдельная копия под бенчмарк
    base_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, base_out)
    del model_B
    torch.cuda.empty_cache()
    return state


def ckpt_name(cfg, tail, noise_sigma, lam, seed):
    return Path(cfg.out_dir) / f"{cfg.dataset}_LateSimkin_{tail}_ns{noise_sigma:g}_lam{lam:g}_s{seed}.pt"


def train_phase3(cfg, base_state, head_state, simk_loader, tr_loader, te_loader, device,
                 tail, noise_sigma, lam, seed):
    """Фаза 3: мультитаск-дообучение хвоста, шум на conv1. Возвращает (model, history, ckpt_path)."""
    print(f"\n[Фаза 3] tail={tail}, шум на conv1 σ={noise_sigma}, λ={lam}, seed={seed}")
    torch.manual_seed(seed)
    model_C = new_model(noise_sigma=noise_sigma).to(device)
    model_C.load_state_dict(base_state)                       # веса базового классификатора
    model_C.simkin_head.load_state_dict(head_state)           # Симкин-голова из фазы 1
    frozen = frozen_pairs(model_C, TAILS[tail])               # soft-freeze фронт-энда (±δ)
    hist = train_classifier(model_C, tr_loader, te_loader, cfg.ft_epochs, cfg.lr, device,
                            multitask=lam > 0, simk_loader=simk_loader, lam=lam,
                            frozen=frozen, delta=cfg.soft_delta, amp=cfg.amp)
    path = ckpt_name(cfg, tail, noise_sigma, lam, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model_C.state_dict(), path)
    return model_C, hist, path


def load_cfg(argv=None):
    """late_simkin (база) <- пресет (напр. exp224) <- CLI key=value."""
    root = OmegaConf.load("params.yaml")
    cli = OmegaConf.from_cli(list(argv if argv is not None else sys.argv[1:]))
    cfg = OmegaConf.merge(root.late_simkin)
    preset = cli.pop("preset", None) if "preset" in cli else None
    if preset:
        cfg = OmegaConf.merge(cfg, root[preset].get("late_simkin", {}))
    cfg = OmegaConf.merge(cfg, cli)
    if cfg.get("render_sigma") is None:
        cfg.render_sigma = root.noise.sigma
    return cfg


def run(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n{OmegaConf.to_yaml(cfg)}")
    simk_loader, tr_loader, te_loader = build_loaders(cfg)
    sigmas = list(cfg.noise_sigmas) if cfg.get("noise_sigmas") else [cfg.noise_sigma]
    results = []
    for seed in cfg.seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        head = get_phase1_head(cfg, simk_loader, device, seed)
        base = get_phase2_base(cfg, tr_loader, te_loader, device, seed)
        for tail in cfg.tails:
            for ns in sigmas:
                if ckpt_name(cfg, tail, ns, cfg.lam, seed).exists() and not cfg.get("overwrite", False):
                    print(f"skip (есть чекпойнт): tail={tail} σ={ns} seed={seed}")
                    continue
                m, hist, path = train_phase3(cfg, base, head, simk_loader, tr_loader, te_loader,
                                             device, tail, ns, cfg.lam, seed)
                results.append(dict(seed=seed, tail=tail, noise_sigma=ns, clean=hist[-1]["test_acc"],
                                    ckpt=str(path)))
                del m
                torch.cuda.empty_cache()
    print("\nИтог (robust считать scripts/bench.py):")
    for r in results:
        print(r)
    return results


if __name__ == "__main__":
    run(load_cfg())
