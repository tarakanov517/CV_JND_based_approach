"""Сравнение Simkin (tile) vs VOne на ОБЩЕЙ оси σ_eff.

Идея: закон шума один и тот же (Poisson≈Gauss, std∝√сигнал), «количество» шума держим
сопоставимым -> ось X это ИЗМЕРЕННЫЙ σ_eff = std инъекции в активации на чистых данных,
а не ручка модели. Для VOne пункт B: под каждый noise_scale модель ДООБУЧЕНА (train_vone),
так что каждая точка обучена-под-свой-шум, как у Simkin — сравнение честное.

Для каждого чекпойнта: sigma_eff, sigma_rel, clean, robust@e (EOT-PGD/PGD), natural-retention.
σ_eff меряется law-agnostic:
    tile  -- разность (вход следующего модуля) − (выход noise-модуля);
    vone  -- разность (выход vone_block с шумом) − (он же без шума).

    python scripts/eval_sigma_eff.py configs/eval_sigma_eff.yaml
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import rootutils
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

rootutils.setup_root(__file__, indicator="src", pythonpath=True)

from src.tile_model import TileResNet
from src.cls_data import get_eval_loader01, load_test_uint8, UInt8Dataset, DATASETS
from src.corruptions import GROUPS, corrupted_images
from scripts.eval_blackbox import Norm01, eot_pgd_acc

NAT = GROUPS["natural"]           # contrast, brightness, fog -- на них виден SNR-коллапс



def build(spec, num_classes, device):
    """(core, Norm01-обёртка[0,1]). arch: 'tile' (Simkin) | 'vone'.
    tile: kwargs TileResNet (tap/noise_tap/dual_bn/noise_sigma/noise_type/noise_scale/noise_level).
    vone: noise_scale/level берутся из сохранённого в ckpt arch, в конфиге хватит name+ckpt+arch."""
    if spec.get("arch", "tile") == "vone":
        from src.vone_model import load_vone_classifier
        core = load_vone_classifier(spec.ckpt, device, noise_on=True, num_classes=num_classes)
        return core, Norm01(core, device).to(device).eval()
    kw = {k: spec[k] for k in ("tap", "noise_tap", "dual_bn", "noise_sigma",
                               "noise_type", "noise_scale", "noise_level") if k in spec}
    core = TileResNet(num_classes=num_classes, **kw).to(device)
    state = torch.load(spec.ckpt, map_location=device)
    core.load_state_dict(state.get("state_dict", state) if isinstance(state, dict) else state)
    core.eval()
    return core, Norm01(core, device).to(device).eval()


@torch.no_grad()
def measure_sigma_eff(core, wrapper, loader01, device, arch="tile", n_batches=3):
    """std инъектированного шума (абс. и относительно активации), среднее по батчам."""
    if arch == "vone":
        vb = core.net.vone_block
        buf = {}
        h = vb.register_forward_hook(lambda m, i, o: buf.__setitem__("o", o.detach()))
        abs_, rel_, nb = 0.0, 0.0, 0
        try:
            for x, _ in loader01:
                if nb >= n_batches:
                    break
                x = x.to(device)
                core.set_noise(False); wrapper(x); clean = buf["o"]
                core.set_noise(True);  wrapper(x); noisy = buf["o"]
                inj = noisy - clean
                abs_ += inj.std().item()
                rel_ += (inj.std() / (clean.abs().mean() + 1e-8)).item()
                nb += 1
        finally:
            h.remove()
            core.set_noise(True)                           # вернуть шум для последующего eval
        return (abs_ / max(nb, 1)), (rel_ / max(nb, 1))

    ni = core._noise_idx                                   # tile: diff вокруг точки впрыска
    pre_mod = core._stages[ni][1]
    post_mod = core._stages[ni + 1][1] if ni + 1 < len(core._stages) else core.class_pool
    buf = {}
    h1 = pre_mod.register_forward_hook(lambda m, i, o: buf.__setitem__("pre", o.detach()))
    h2 = post_mod.register_forward_pre_hook(lambda m, a: buf.__setitem__("post", a[0].detach()))
    abs_, rel_, nb = 0.0, 0.0, 0
    try:
        for x, _ in loader01:
            if nb >= n_batches:
                break
            wrapper(x.to(device))
            inj = buf["post"] - buf["pre"]
            abs_ += inj.std().item()
            rel_ += (inj.std() / (buf["pre"].abs().mean() + 1e-8)).item()
            nb += 1
    finally:
        h1.remove()
        h2.remove()
    return (abs_ / max(nb, 1)), (rel_ / max(nb, 1))


# ───────────────────────────── accuracy ─────────────────────────────

@torch.no_grad()
def accuracy(wrapper, loader, device, reps=1):
    c = t = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = sum(wrapper(x).softmax(1) for _ in range(reps)) / reps
        c += (logits.argmax(1) == y).sum().item()
        t += y.size(0)
    return c / t


def _knob(spec, core):
    if spec.get("arch", "tile") == "vone":
        return float(core.arch.get("noise_scale", 0.0))
    return float(spec.get("noise_sigma", spec.get("noise_scale", 0.0)))


# ───────────────────────────── main ─────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("overrides", nargs="*", help="OmegaConf key=value")
    a = ap.parse_args()
    cfg = OmegaConf.merge(OmegaConf.load(a.config), OmegaConf.from_cli(a.overrides))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = DATASETS[cfg.dataset]["num_classes"]
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    eps_list = list(cfg.eps_255)
    print(f"Device: {device} | dataset={cfg.dataset} | models={[m.name for m in cfg.models]}")

    atk_ld = get_eval_loader01(cfg.dataset, cfg.n, cfg.seed_eval, cfg.batch_size,
                               cfg.num_workers, cfg.data_root)
    images, labels = load_test_uint8(cfg.dataset, cfg.data_root, cfg.cache_dir)
    if cfg.n is not None and cfg.n < len(labels):
        g = torch.Generator().manual_seed(cfg.seed_eval)
        idx = torch.randperm(len(labels), generator=g)[:cfg.n].numpy()
        images, labels = images[idx], labels[idx]
    clean_ld = DataLoader(UInt8Dataset(images, labels), batch_size=256, num_workers=0)

    rows = []
    for spec in cfg.models:
        arch = spec.get("arch", "tile")
        print(f"\n── {spec.name} ({arch}) ──")
        core, m = build(spec, num_classes, device)
        s_abs, s_rel = measure_sigma_eff(core, m, atk_ld, device, arch, cfg.get("sigma_batches", 3))
        stoch = s_abs > 1e-4
        k = int(cfg.eot_k) if stoch else 0
        reps = int(cfg.get("vote_reps", 8)) if stoch else 1

        row = dict(model=spec.name, arch=arch,
                   noise_type=("neuronal" if arch == "vone" else spec.get("noise_type", "additive")),
                   knob=_knob(spec, core), sigma_eff=round(s_abs, 4), sigma_rel=round(s_rel, 4))

        clean_sub = accuracy(m, clean_ld, device, reps)
        row["clean"] = round(clean_sub, 4)
        print(f"   σ_eff={s_abs:.4f}  σ_rel={s_rel:.4f}  clean={clean_sub:.4f}  (stoch={stoch}, k={k}, reps={reps})")

        for e in eps_list:
            r = eot_pgd_acc(m, atk_ld, e / 255, device, steps=cfg.steps, eot=k, reps=reps, amp=cfg.amp)
            row[f"robust@{e}"] = round(r, 4)
            print(f"   robust@{e}/255 = {r:.4f}")

        for c in NAT:
            accs = []
            for s in cfg.severities:
                cache = Path(cfg.cache_dir) / "corrupt" / f"{cfg.dataset}_n{len(labels)}" / f"{c}_s{s}.npy"
                imgs_c = corrupted_images(images, c, s, cache, n_jobs=cfg.get("n_jobs", -1))
                ld = DataLoader(UInt8Dataset(imgs_c, labels), batch_size=256, num_workers=0)
                accs.append(accuracy(m, ld, device, reps))
            ret = float(np.mean(accs)) / clean_sub if clean_sub > 0 else float("nan")
            row[c] = round(ret, 4)
            print(f"   {c:12s} retention = {ret:.4f}")
        row["natural_avg"] = round(float(np.mean([row[c] for c in NAT])), 4)

        rows.append(row)
        pd.DataFrame(rows).to_csv(out / "sigma_eff_eval.csv", index=False)
        del core, m
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows).sort_values("sigma_eff")
    print("\n" + "=" * 70)
    print(df.to_string(index=False))
    df.to_csv(out / "sigma_eff_eval.csv", index=False)
    print(f"\n→ {out / 'sigma_eff_eval.csv'}")


if __name__ == "__main__":
    main()