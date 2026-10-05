"""
Секции (--only a,b,...; по умолчанию все):
    clean       -- clean accuracy на всём тесте (для шумных моделей: 1 проход, как раньше, + голосование reps)
    whitebox    -- EOT-PGD (k для стохастических, обычный PGD для детерминированных), eps-список
    corruptions -- ImageNet-C протокол (natural / digital / noise), retention = acc/clean
Всё пишется в {out_dir}/{section}.csv (резюмируемо: посчитанные ключи пропускаются) + summary.md.

    python scripts/bench.py configs/bench_imagenette.yaml
    python scripts/bench.py configs/bench_imagenette.yaml --only clean,whitebox
    python scripts/bench.py configs/bench_stl10.yaml             # те же метрики на 96 (контроль)
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

from src.cls_data import get_eval_loader01, load_test_uint8, UInt8Dataset
from src.corruptions import ALL, GROUPS, corrupted_images, group_of
from scripts.eval_blackbox import load_model, eot_pgd_acc, square_acc, craft_transfer, acc_on

SECTIONS = ["clean", "whitebox", "corruptions"]


# ───────────────────────────── models ─────────────────────────────

def is_stochastic(spec):
    if spec.arch == "vone":
        return bool(spec.get("noise_on", True))
    return float(spec.get("noise_sigma", 0.0)) > 0


def build(spec, device):
    if spec.arch == "vone":
        return load_model(spec.ckpt, 1.0 if spec.get("noise_on", True) else 0.0, device, arch="vone")
    return load_model(spec.ckpt, float(spec.get("noise_sigma", 0.0)), device, arch=spec.arch,
                      tap=spec.get("tap", "l4"), noise_tap=spec.get("noise_tap", "conv1"),
                      dual_bn=spec.get("dual_bn", True))


# ───────────────────────────── io helpers ─────────────────────────────

class Table:
    def __init__(self, path: Path, keys):
        self.path, self.keys = path, keys
        self.rows = pd.read_csv(path).to_dict("records") if path.exists() else []

    def has(self, **kv):
        return any(all(str(r.get(k)) == str(v) for k, v in kv.items()) for r in self.rows)

    def add(self, **row):
        self.rows.append(row)
        pd.DataFrame(self.rows).to_csv(self.path, index=False)
        print("   ", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})

    def df(self):
        return pd.DataFrame(self.rows)


@torch.no_grad()
def accuracy(model01, loader, device, reps=1, amp=False):
    use_amp = amp and device.type == "cuda"
    c = t = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            logits = sum(model01(x).float().softmax(1) for _ in range(reps)) / reps
        c += (logits.argmax(1) == y).sum().item()
        t += y.size(0)
    return c / t


# ───────────────────────────── sections ─────────────────────────────

def sec_clean(cfg, models, device, out):
    T = Table(out / "clean.csv", ["model"])
    ld = get_eval_loader01(cfg.dataset, None, 0, 256, cfg.num_workers, cfg.data_root)
    for spec in models:
        if T.has(model=spec.name):
            continue
        m = build(spec, device)
        reps = cfg.clean.vote_reps if is_stochastic(spec) else 1
        T.add(model=spec.name, clean=accuracy(m, ld, device, 1, cfg.amp),
              clean_vote=accuracy(m, ld, device, reps, cfg.amp), vote_reps=reps)
        del m
    return T.df()


def sec_whitebox(cfg, models, device, out):
    W = cfg.whitebox
    T = Table(out / "whitebox.csv", ["model", "eps"])
    ld = get_eval_loader01(cfg.dataset, W.n, cfg.seed_eval, W.batch_size, cfg.num_workers, cfg.data_root)
    for spec in models:
        k = W.eot_k if is_stochastic(spec) else 0
        m = None
        for e in W.eps_255:
            if T.has(model=spec.name, eps=e):
                continue
            m = m if m is not None else build(spec, device)
            if not T.has(model=spec.name, eps=0):
                T.add(model=spec.name, eps=0, robust=accuracy(m, ld, device, 1, cfg.amp),
                      attack="none", eot_k=0, n=len(ld.dataset))
            T.add(model=spec.name, eps=e, robust=eot_pgd_acc(m, ld, e / 255, device, W.steps, k, amp=cfg.amp),
                  attack="EOT-PGD" if k else "PGD", eot_k=k, n=len(ld.dataset))
        del m
    return T.df()

def sec_corruptions(cfg, models, device, out):
    C = cfg.corruptions
    names = ALL if C.names == "all" else list(C.names)
    T = Table(out / "corruptions.csv", ["model", "corruption", "severity"])
    images, labels = load_test_uint8(cfg.dataset, cfg.data_root, cfg.cache_dir)
    if C.n is not None and C.n < len(labels):
        g = torch.Generator().manual_seed(cfg.seed_eval)
        idx = torch.randperm(len(labels), generator=g)[:C.n].numpy()
        images, labels = images[idx], labels[idx]
    clean_ld = DataLoader(UInt8Dataset(images, labels), batch_size=256, num_workers=0)
    built = {}

    def get(spec):
        if spec.name not in built:
            built.clear()
            torch.cuda.empty_cache()
            built[spec.name] = build(spec, device)
        return built[spec.name]

    for spec in models:
        if not T.has(model=spec.name, corruption="clean", severity=0):
            T.add(model=spec.name, corruption="clean", group="clean", severity=0,
                  acc=accuracy(get(spec), clean_ld, device, C.reps, cfg.amp))
    for c in names:
        for s in C.severities:
            need = [spec for spec in models if not T.has(model=spec.name, corruption=c, severity=s)]
            if not need:
                continue
            use_cache = bool(C.get("cache", False))
            cache = Path(cfg.cache_dir) / "corrupt" / f"{cfg.dataset}_n{len(labels)}" / f"{c}_s{s}.npy" if use_cache else None
            imgs_c = corrupted_images(images, c, s, cache, n_jobs=C.n_jobs)
            ld = DataLoader(UInt8Dataset(imgs_c, labels), batch_size=256, num_workers=0)
            for spec in need:
                T.add(model=spec.name, corruption=c, group=group_of(c), severity=s,
                      acc=accuracy(get(spec), ld, device, C.reps, cfg.amp))
    return T.df()


# ───────────────────────────── summary ─────────────────────────────

def summarize(out: Path, order):
    lines = []
    fmt = lambda d: d.to_markdown(floatfmt=".3f") if hasattr(d, "to_markdown") else d.to_string()
    order_idx = lambda d: d.reindex([m for m in order if m in d.index])
    try:
        if (out / "whitebox.csv").exists():
            w = pd.read_csv(out / "whitebox.csv")
            p = w.pivot_table(index="model", columns="eps", values="robust")
            p.columns = ["clean(subset)" if c == 0 else f"@{c}/255" for c in p.columns]
            lines += ["## White-box (EOT-PGD / PGD)", fmt(order_idx(p)), ""]
        if (out / "clean.csv").exists():
            lines += ["## Clean (весь тест)", fmt(order_idx(pd.read_csv(out / "clean.csv").set_index("model"))), ""]
        if (out / "corruptions.csv").exists():
            c = pd.read_csv(out / "corruptions.csv")
            clean = c[c.corruption == "clean"].set_index("model").acc
            c = c[c.corruption != "clean"].copy()
            c["retention"] = c.acc / c.model.map(clean)
            per_c = c.groupby(["model", "corruption"]).retention.mean().unstack().copy()
            cols = []
            for g, cs in GROUPS.items():
                have = [x for x in cs if x in per_c.columns]
                if have:
                    per_c[f"{g}_avg"] = per_c[have].mean(1)
                    cols += have + [f"{g}_avg"]
            lines += ["## Corruptions: retention = acc/clean (среднее по severity)", fmt(order_idx(per_c[cols])), ""]
    except Exception as ex:
        lines.append(f"(summary error: {ex})")
    (out / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--only", default=",".join(SECTIONS))
    ap.add_argument("overrides", nargs="*", help="OmegaConf key=value")
    a = ap.parse_args()
    cfg = OmegaConf.merge(OmegaConf.load(a.config), OmegaConf.from_cli(a.overrides))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    models = [m for m in cfg.models if m.get("enabled", True)]
    print(f"Device: {device} | dataset={cfg.dataset} | models={[m.name for m in models]}")
    for sec in a.only.split(","):
        print(f"\n══ {sec} ══")
        globals()[f"sec_{sec}"](cfg, models, device, out)
    summarize(out, [m.name for m in cfg.models])


if __name__ == "__main__":
    main()
