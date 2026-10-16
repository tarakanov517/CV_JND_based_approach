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


# ───────────────────────────── build ─────────────────────────────

def build(spec, num_classes, device):
    kw = {k: spec[k] for k in ("tap", "noise_tap", "dual_bn", "noise_sigma",
                               "noise_type", "noise_scale", "noise_level") if k in spec}
    core = TileResNet(num_classes=num_classes, **kw).to(device)
    state = torch.load(spec.ckpt, map_location=device)
    core.load_state_dict(state.get("state_dict", state) if isinstance(state, dict) else state)
    core.eval()
    return core, Norm01(core, device).to(device).eval()


# ───────────────────────────── σ_eff ─────────────────────────────

@torch.no_grad()
def measure_sigma_eff(core, wrapper, loader01, device, n_batches=3):
    ni = core._noise_idx
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
            wrapper(x.to(device))                      # полный eval-проход (шум включён)
            inj = buf["post"] - buf["pre"]             # впрыснутый шум
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

    # данные
    atk_ld = get_eval_loader01(cfg.dataset, cfg.n, cfg.seed_eval, cfg.batch_size,
                               cfg.num_workers, cfg.data_root)                      # [0,1] для σ_eff/атак
    images, labels = load_test_uint8(cfg.dataset, cfg.data_root, cfg.cache_dir)      # uint8 для коррапшенов
    if cfg.n is not None and cfg.n < len(labels):
        g = torch.Generator().manual_seed(cfg.seed_eval)
        idx = torch.randperm(len(labels), generator=g)[:cfg.n].numpy()
        images, labels = images[idx], labels[idx]
    clean_ld = DataLoader(UInt8Dataset(images, labels), batch_size=256, num_workers=0)

    rows = []
    for spec in cfg.models:
        print(f"\n── {spec.name} ──")
        core, m = build(spec, num_classes, device)
        s_abs, s_rel = measure_sigma_eff(core, m, atk_ld, device, cfg.get("sigma_batches", 3))
        stoch = s_abs > 1e-4
        k = int(cfg.eot_k) if stoch else 0
        reps = int(cfg.get("vote_reps", 8)) if stoch else 1

        row = dict(model=spec.name, noise_type=spec.get("noise_type", "additive"),
                   knob=float(spec.get("noise_sigma", spec.get("noise_scale", 0.0))),
                   sigma_eff=round(s_abs, 4), sigma_rel=round(s_rel, 4))

        clean_sub = accuracy(m, clean_ld, device, reps)                 # знаменатель retention
        row["clean"] = round(clean_sub, 4)
        print(f"   σ_eff={s_abs:.4f}  σ_rel={s_rel:.4f}  clean={clean_sub:.4f}  (stoch={stoch}, k={k}, reps={reps})")

        for e in eps_list:                                              # adversarial
            r = eot_pgd_acc(m, atk_ld, e / 255, device, steps=cfg.steps, eot=k, reps=reps, amp=cfg.amp)
            row[f"robust@{e}"] = round(r, 4)
            print(f"   robust@{e}/255 = {r:.4f}")

        for c in NAT:                                                   # natural-коррапшены (механизм)
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