"""Задача 1. Noise-ablation для late-simkin (голова l4, шум на conv1), по умолчанию STL-10 / 96.

Идём по σ по возрастанию (от 0 и текущей 0.5 вверх), пока модель не «сломается»:
    bad(σ) = clean < clean_baseline - clean_drop   или   train_acc(последняя эпоха) < min_train_acc

На каждую σ пишется строка в CSV (резюмируемо: готовые строки пропускаются):
    clean, robust@eps (EOT-PGD k для σ>0, PGD для σ=0), train_acc, act_std на conv1, σ/act_std
В конце: проверка EOT по k (2·k) на максимальной «хорошей» σ -- исключаем маскировку градиента
при большом шуме, и подсказка σ* (max robust@eps2 при clean ≥ baseline - pick_clean_drop).

Запуск:
    python scripts/noise_ablation.py
    python scripts/noise_ablation.py ab.sigmas=[0.5,1,2,4] ab.seeds=[42,0] ab.with_no_simkin=true
    python scripts/noise_ablation.py ab.eval_n=null          # EOT-PGD на полном тесте (дорого)
Ключи `ab.*` -> секция noise_ablation, остальные -> late_simkin (как в src/train_late_simkin.py).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import rootutils
from omegaconf import OmegaConf

rootutils.setup_root(__file__, indicator="src", pythonpath=True)

from src.cls_data import get_eval_loader01
from src.train_late_simkin import (build_loaders, get_phase1_head, get_phase2_base, train_phase3,
                                   new_model, ckpt_name, load_cfg)
from scripts.eval_blackbox import Norm01, eot_pgd_acc

COLS = ["seed", "config", "noise_sigma", "lam", "clean", "train_acc", "act_std", "sigma_over_act",
        "attack", "eot_k", "bad", "ckpt"]


def parse_args(argv):
    ab_args = [a for a in argv if a.startswith("ab.")]
    ls_args = [a for a in argv if not a.startswith("ab.")]
    root = OmegaConf.load("params.yaml")
    ab = OmegaConf.merge(root.noise_ablation, OmegaConf.from_cli([a[3:] for a in ab_args]))
    cfg = load_cfg(ls_args)
    return cfg, ab


@torch.no_grad()
def act_stats(model, loader01, device, n_batches=4):
    """std активаций в точке впрыска шума (до шума) на чистом тесте -> масштаб для σ."""
    _, mod = model._stages[model._noise_idx]
    acts = []
    h = mod.register_forward_hook(lambda m, i, o: acts.append(o.detach().float().flatten().cpu()))
    m01 = Norm01(model, device).eval()
    for i, (x, _) in enumerate(loader01):
        if i >= n_batches:
            break
        m01(x.to(device))
    h.remove()
    a = torch.cat(acts)
    return float(a.std())


@torch.no_grad()
def clean_acc(model, loader01, device, reps=1):
    m01 = Norm01(model, device).eval()
    c = t = 0
    for x, y in loader01:
        x, y = x.to(device), y.to(device)
        logits = sum(m01(x) for _ in range(reps)) / reps
        c += (logits.argmax(1) == y).sum().item()
        t += y.size(0)
    return c / t


def evaluate(model, ab, loaders, device, noise_sigma):
    full01, sub01 = loaders
    model.eval()
    k = ab.eot_k if noise_sigma > 0 else 0
    res = dict(clean=clean_acc(model, full01, device, ab.clean_reps),
               act_std=act_stats(model, full01, device),
               attack="EOT-PGD" if k else "PGD", eot_k=k)
    res["sigma_over_act"] = noise_sigma / max(res["act_std"], 1e-8)
    m01 = Norm01(model, device).eval()
    for e in ab.eps_255:
        res[f"eps{e}"] = eot_pgd_acc(m01, sub01, e / 255, device, steps=ab.pgd_steps, eot=k)
        print(f"    eps={e}/255 {res['attack']}(k={k}) = {res[f'eps{e}']:.4f}")
    return res


def main(argv=None):
    cfg, ab = parse_args(list(argv if argv is not None else sys.argv[1:]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n[late_simkin]\n{OmegaConf.to_yaml(cfg)}[noise_ablation]\n{OmegaConf.to_yaml(ab)}")

    out_csv = Path(ab.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(out_csv) if out_csv.exists() else pd.DataFrame(columns=COLS)
    rows = df.to_dict("records")

    def done(seed, config, ns, lam):
        return next((r for r in rows if r["seed"] == seed and r["config"] == config
                     and np.isclose(r["noise_sigma"], ns) and np.isclose(r["lam"], lam)), None)

    def flush():
        pd.DataFrame(rows).to_csv(out_csv, index=False)

    simk_loader, tr_loader, te_loader = build_loaders(cfg)
    full01 = get_eval_loader01(cfg.dataset, n=None, batch_size=256, num_workers=cfg.num_workers,
                               root=cfg.data_root)
    sub01 = get_eval_loader01(cfg.dataset, n=ab.eval_n, seed=ab.eval_seed, batch_size=128,
                              num_workers=cfg.num_workers, root=cfg.data_root)
    sigmas = sorted(float(s) for s in ab.sigmas)
    lams = [float(ab.lam)] + ([0.0] if ab.with_no_simkin and ab.lam > 0 else [])

    for seed in ab.seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        head = get_phase1_head(cfg, simk_loader, device, seed)
        base = get_phase2_base(cfg, tr_loader, te_loader, device, seed)

        # baseline (фаза 2, без шума) -> опорный clean
        r0 = done(seed, "baseline", 0.0, 0.0)
        if r0 is None:
            print(f"\n[seed {seed}] оценка baseline")
            m = new_model(0.0).to(device)
            m.load_state_dict(base)
            r0 = dict(seed=seed, config="baseline", noise_sigma=0.0, lam=0.0, train_acc=np.nan, bad=False,
                      ckpt=f"{cfg.out_dir}/{cfg.dataset}_baseline_s{seed}.pt",
                      **evaluate(m, ab, (full01, sub01), device, 0.0))
            rows.append(r0)
            flush()
            del m
        clean_ref = r0["clean"]
        print(f"[seed {seed}] baseline clean = {clean_ref:.4f}")

        for lam in lams:
            config = "late_simkin" if lam > 0 else "noise_only"
            n_bad = 0
            for ns in sigmas:
                r = done(seed, config, ns, lam)
                if r is None:
                    tail = ab.tail
                    path = ckpt_name(cfg, tail, ns, lam, seed)
                    hist_path = path.with_suffix(".hist.json")
                    if path.exists():
                        m = new_model(ns).to(device)
                        m.load_state_dict(torch.load(path, map_location=device))
                        hist = json.loads(hist_path.read_text()) if hist_path.exists() else [{}]
                    else:
                        m, hist, path = train_phase3(cfg, base, head, simk_loader, tr_loader, te_loader,
                                                     device, tail, ns, lam, seed)
                        hist_path.write_text(json.dumps(hist))
                    print(f"  оценка σ={ns} λ={lam}")
                    ev = evaluate(m, ab, (full01, sub01), device, ns)
                    tr_acc = hist[-1].get("train_acc", np.nan)
                    bad = bool(ev["clean"] < clean_ref - ab.clean_drop
                               or (tr_acc == tr_acc and tr_acc < ab.min_train_acc))
                    r = dict(seed=seed, config=config, noise_sigma=ns, lam=lam, train_acc=tr_acc,
                             bad=bad, ckpt=str(path), **ev)
                    rows.append(r)
                    flush()
                    del m
                    torch.cuda.empty_cache()
                e0 = ab.eps_255[0]
                print(f"  [{config} s{seed}] σ={ns:<5g} clean={r['clean']:.4f} "
                      f"eps{e0}={r[f'eps{e0}']:.4f} σ/act={r['sigma_over_act']:.2f} bad={r['bad']}")
                n_bad = n_bad + 1 if r["bad"] else 0
                if n_bad >= ab.patience:
                    print(f"  стоп: {ab.patience} подряд bad-точек (последняя σ={ns})")
                    break

    # проверка EOT по k на максимальной хорошей σ (на seed[0], основная конфигурация)
    seed = ab.seeds[0]
    good = [r for r in rows if r["seed"] == seed and r["config"] == "late_simkin" and not r["bad"]
            and r["noise_sigma"] > 0]
    if good:
        g = max(good, key=lambda r: r["noise_sigma"])
        if done(seed, "k_check", g["noise_sigma"], g["lam"]) is None:
            k2 = 2 * ab.eot_k
            print(f"\n[k-check] σ={g['noise_sigma']} EOT k={ab.eot_k} -> {k2}")
            m = new_model(g["noise_sigma"]).to(device)
            m.load_state_dict(torch.load(g["ckpt"], map_location=device))
            e0 = ab.eps_255[0]
            rk = eot_pgd_acc(Norm01(m, device).eval(), sub01, e0 / 255, device, steps=ab.pgd_steps, eot=k2)
            rows.append(dict(seed=seed, config="k_check", noise_sigma=g["noise_sigma"], lam=g["lam"],
                             clean=g["clean"], attack="EOT-PGD", eot_k=k2, bad=False, ckpt=g["ckpt"],
                             **{f"eps{e0}": rk}))
            flush()
            print(f"  eps{e0}: k={ab.eot_k} {g[f'eps{e0}']:.4f} | k={k2} {rk:.4f} "
                  f"(падение > ~0.02 -> подозрение на маскировку, поднимать k)")

    report(pd.DataFrame(rows), ab, out_csv)


def report(df, ab, out_csv):
    e0 = ab.eps_255[0]
    main_df = df[df.config.isin(["baseline", "late_simkin", "noise_only"])]
    agg = (main_df.groupby(["config", "noise_sigma"])
           .agg(clean=("clean", "mean"), **{f"eps{e}": (f"eps{e}", "mean") for e in ab.eps_255},
                sigma_over_act=("sigma_over_act", "mean"), n_seeds=("seed", "nunique"), bad=("bad", "max"))
           .reset_index())
    print("\n" + agg.to_string(index=False))
    agg.to_csv(out_csv.with_name(out_csv.stem + "_agg.csv"), index=False)

    ls = agg[agg.config == "late_simkin"]
    ref = agg[agg.config == "baseline"].clean.mean()
    ok = ls[ls.clean >= ref - ab.get("pick_clean_drop", 0.15)]
    if len(ok):
        best = ok.loc[ok[f"eps{e0}"].idxmax()]
        print(f"\nσ* (max eps{e0} при clean ≥ baseline-{ab.get('pick_clean_drop', 0.15)}): "
              f"σ={best.noise_sigma:g} clean={best.clean:.4f} eps{e0}={best[f'eps{e0}']:.4f}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for cfgname, ls_ in (("late_simkin", "-"), ("noise_only", "--")):
            d = agg[agg.config == cfgname].sort_values("noise_sigma")
            if not len(d):
                continue
            ax.plot(d.noise_sigma, d.clean, "o" + ls_, color="#2ca02c", label=f"clean · {cfgname}")
            for e, col in zip(ab.eps_255, ["#1f77b4", "#d62728", "#9467bd"]):
                ax.plot(d.noise_sigma, d[f"eps{e}"], "o" + ls_, color=col, label=f"robust @{e}/255 · {cfgname}")
        ax.axhline(ref, color="#8c8c8c", lw=.8, ls=":", label="baseline clean")
        ax.axhline(ref - ab.clean_drop, color="#8c8c8c", lw=.8, ls="-.", label="порог «плохо учится»")
        ax.set_xscale("symlog", linthresh=0.25)
        ax.set_xlabel("σ шума на conv1"); ax.set_ylabel("accuracy"); ax.set_ylim(0, 1)
        ax.set_title("Noise-ablation, late-simkin"); ax.grid(alpha=.3); ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out_csv.with_suffix(".png"), dpi=130)
        print(f"график: {out_csv.with_suffix('.png')}")
    except Exception as ex:                         
        print(f"(график не построен: {ex})")


if __name__ == "__main__":
    main()
