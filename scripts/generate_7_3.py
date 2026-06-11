"""
Генерация датасета изображений эксперимента 7.3 (Симкин).

Запуск:
    python scripts/generate_7_3.py
"""

import csv
import random
import os
import cv2
import rootutils
import numpy as np
from joblib import delayed, Parallel
from omegaconf import OmegaConf, DictConfig
from pathlib import Path
from tqdm import tqdm

rootutils.setup_root(__file__, indicator="src", pythonpath=True)

from src.display import phi0_arcmin
from src.stimulus import make_luminance_map, luminance_map_to_canvas, deformation, add_noise
from src.common import make_dir
from src.visibility_funcs import is_visible_7_3

COUNTS = {
    "niche": 100,
    "triangle": 100,
    "rect": 100,
    "circle": 100,
}

THETA_CORNER = list(range(1, 17))
THETA_NICHE  = list(range(1, 33))
SIDES_NICHE  = ["top", "bottom", "left", "right"]


def sample_luminance(cfg: DictConfig) -> DictConfig:
    d = cfg.display
    L_outer = float(np.exp(random.uniform(np.log(d.min_brightness + 0.01), np.log(d.max_brightness))))
    dL = random.uniform(0.01, 1.5) * L_outer
    L_inner = float(np.clip(L_outer - dL, d.min_brightness, d.max_brightness))
    return OmegaConf.merge(cfg, {"luminance": {"outer": L_outer, "inner": L_inner}})


def process_one(task: dict, out_dir: Path) -> dict:
    cfg = task["cfg"]
    mode = task["mode"]
    params = task["params"]
    seed = task["seed"]
    label = task["label"]
    idx = task["idx"]

    np.random.seed(seed)
    square = "left" if np.random.random() > 0.5 else "right"
    lmap = make_luminance_map(cfg)
    canvas = luminance_map_to_canvas(lmap, cfg)
    deformation(canvas, cfg, mode=mode, params=params, square=square)
    canvas = add_noise(canvas, sigma=cfg.noise.sigma)

    filename = f"{label}_{idx:04d}.png"
    cv2.imwrite(str(out_dir / filename), cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR))

    return {
        "filename": filename,
        "mode": mode,
        "params": str(params),
        "L_outer": round(float(cfg.luminance.outer), 2),
        "L_inner": round(float(cfg.luminance.inner), 2),
        "visible": is_visible_7_3(mode, params, cfg),
    }


def build_tasks(cfg: DictConfig) -> list[dict]:
    random.seed(cfg.general.seed)
    tasks = []
    global_idx = 0
    for mode, count in COUNTS.items():
        for _ in range(count):
            sample_cfg = sample_luminance(cfg)
            if mode == "niche":
                side = random.choice(SIDES_NICHE)
                h = random.choice(THETA_NICHE)
                w = random.choice(THETA_NICHE)
                params = (side, h, w)
                label = f"niche_{side}_h{h:02d}_w{w:02d}"
            else:
                theta = random.choice(THETA_CORNER)
                params = theta
                label = f"{mode}_t{theta:02d}"

            tasks.append({
                "cfg": sample_cfg,
                "mode": mode,
                "params": params,
                "label": label,
                "seed": cfg.general.seed + global_idx,
                "idx": global_idx,
            })
            global_idx += 1
    return tasks


def main(cfg: DictConfig):
    out_dir = make_dir(Path(cfg.output.path).parent / "dataset_7_3", delete_if_exist=True)
    csv_path = Path(cfg.output.path).parent / "labels_7_3.csv"

    print(f"φ₀ = {phi0_arcmin(cfg):.3f} arcmin/pixel")
    
    tasks = build_tasks(cfg)

    n_jobs = os.cpu_count()
    rows = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(process_one)(task, out_dir)
        for task in tqdm(tasks)
    )

    fieldnames = ["filename", "mode", "params", "L_outer", "L_inner", "visible"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    cfg = OmegaConf.load("params.yaml")
    main(cfg)
