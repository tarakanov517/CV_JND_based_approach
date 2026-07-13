import csv
import os
import random
from pathlib import Path

import cv2
import numpy as np
import rootutils
from joblib import Parallel, delayed
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

rootutils.setup_root(__file__, indicator="src", pythonpath=True)

from src.common import make_dir
from src.display import cd_to_pixel
from src.stimulus import add_overshoots, luminance_map_to_canvas
from src.visibility_funcs import is_visible_8_1


TILE_SIZE = 128         # масштаб эксперимента
PHI_A_VALUES = [1, 2, 4, 8]   # размеры объекта
PHI_V = 4               # ширина кольца выброса
FIELD_SIZE = 19         # центральный квадрат L_N

EPSILON_VALUES = [0, 4, 16, 32]
SLOPE_FORMS = ['sharp', 'hyperbolic', 'exponential', 'gaussian', 'triangle', 'cosine']
SAMPLES_PER_COMBO = 50
PSI_MIN, PSI_MAX = 0.2, 5.0
L_MIN = 3.0                     # нижняя граница яркости



def signal_energy(cfg: DictConfig, L_N: float, L_obj: float,
                  phi_a: int, epsilon: float, slope: str) -> float:
    pv = PHI_V
    size = phi_a + 2 * pv
    patch = np.full((size, size), L_N, dtype=np.float32)
    patch[pv:pv + phi_a, pv:pv + phi_a] = L_obj
    if epsilon > 0:
        add_overshoots(patch, pv, pv, phi_a, pv, epsilon,
                       abs(L_N - L_obj), slope_form=slope)
    canvas = luminance_map_to_canvas(patch, cfg)
    field_p = float(cd_to_pixel(L_N, cfg))
    s = canvas.astype(np.float32) - field_p
    return float(np.sqrt(np.sum(s * s)))


def solve_L_obj(cfg: DictConfig, L_N: float, target_energy: float,
                phi_a: int, epsilon: float, slope: str) -> float:
    """Бисекцией подбирает L_obj (темнее L_N), чтобы энергия примерно равнялась target_energy."""
    Lmin = float(cfg.display.min_brightness)
    dL_max = float(L_N) - Lmin
    if dL_max <= 0:
        return float(L_N)
    if signal_energy(cfg, L_N, Lmin, phi_a, epsilon, slope) <= target_energy:
        return Lmin
    lo, hi = 0.0, dL_max
    for _ in range(25):
        mid = 0.5 * (lo + hi)
        e = signal_energy(cfg, L_N, L_N - mid, phi_a, epsilon, slope)
        if e < target_energy:
            lo = mid
        else:
            hi = mid
    return float(np.clip(L_N - 0.5 * (lo + hi), Lmin, L_N))


def make_tile_lmap(L_a: float, L_N: float, L_obj: float,
                   phi_a: int, epsilon: float, slope: str):

    lmap = np.full((TILE_SIZE, TILE_SIZE), L_a, dtype=np.float32)   # адаптация

    fo = (TILE_SIZE - FIELD_SIZE) // 2                              # поле L_N
    lmap[fo:fo + FIELD_SIZE, fo:fo + FIELD_SIZE] = L_N

    ix = iy = (TILE_SIZE - phi_a) // 2                              # объект в центре
    lmap[iy:iy + phi_a, ix:ix + phi_a] = L_obj

    if epsilon > 0:                                                 # выброс в кольце
        add_overshoots(lmap, ix, iy, phi_a, PHI_V, epsilon,
                       abs(L_N - L_obj), slope_form=slope)

    return lmap, ix, iy


def process_one(task: dict, out_dir: Path) -> dict:
    cfg = task["cfg"]
    phi_a, epsilon, slope = task["phi_a"], task["epsilon"], task["slope"]
    L_a, L_N, psi, idx = task["L_a"], task["L_N"], task["psi"], task["idx"]

    thr = float(cfg.visual.d_prime) * float(cfg.noise.sigma)        # порог энергии (ψ=1)
    L_obj = solve_L_obj(cfg, L_N, psi * thr, phi_a, epsilon, slope)

    lmap, ix, iy = make_tile_lmap(L_a, L_N, L_obj, phi_a, epsilon, slope)
    canvas_clean = luminance_map_to_canvas(lmap, cfg)              # до шума -- для детектора

    sample_cfg = OmegaConf.merge(cfg, {"luminance": {"outer": L_N, "inner": L_obj}})

    label = f"phi{phi_a:02d}_eps{epsilon:02d}_{slope}"
    filename = f"{label}_{idx:05d}.png"
    cv2.imwrite(str(out_dir / filename),
                cv2.cvtColor(canvas_clean, cv2.COLOR_GRAY2BGR))

    return {
        "filename": filename,
        "phi_a": phi_a,
        "epsilon": epsilon,
        "slope": slope,
        "phi_v": PHI_V,
        "field_size": FIELD_SIZE,
        "tile_size": TILE_SIZE,
        "psi": round(psi, 3),
        "L_a": round(float(L_a), 2),
        "L_N": round(float(L_N), 2),
        "L_obj": round(float(L_obj), 2),
        "visible": is_visible_8_1(canvas_clean, ix, iy, phi_a, PHI_V, sample_cfg),
    }


def build_tasks(cfg: DictConfig) -> list[dict]:
    random.seed(cfg.general.seed)
    d = cfg.display
    tasks, idx = [], 0

    for phi_a in PHI_A_VALUES:
        for epsilon in EPSILON_VALUES:
            slopes = SLOPE_FORMS if epsilon > 0 else [SLOPE_FORMS[0]]
            for slope in slopes:
                for _ in range(SAMPLES_PER_COMBO):
                    psi = float(np.exp(random.uniform(np.log(PSI_MIN), np.log(PSI_MAX))))
                    L_a = float(np.exp(random.uniform(np.log(L_MIN), np.log(d.max_brightness))))
                    L_N = float(np.exp(random.uniform(np.log(L_MIN), np.log(d.max_brightness))))
                    tasks.append({
                        "cfg": cfg, "phi_a": phi_a, "epsilon": epsilon, "slope": slope,
                        "psi": round(psi, 3), "L_a": L_a, "L_N": L_N, "idx": idx,
                    })
                    idx += 1
    return tasks


def main(cfg: DictConfig):
    out_dir = make_dir(Path(cfg.general.output_folder) / "dataset_tile", delete_if_exist=True)
    csv_path = Path(cfg.general.output_folder) / "labels_tile.csv"

    tasks = build_tasks(cfg)
    rows = Parallel(n_jobs=os.cpu_count(), backend="threading")(
        delayed(process_one)(task, out_dir) for task in tqdm(tasks)
    )

    fieldnames = ["filename", "phi_a", "epsilon", "slope", "phi_v", "field_size",
                  "tile_size", "psi", "L_a", "L_N", "L_obj", "visible"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    n_vis = sum(r["visible"] for r in rows)
    print(f"\nСгенерировано {len(rows)} тайлов {TILE_SIZE}×{TILE_SIZE} → {out_dir}")
    print(f"visible=1: {n_vis}/{len(rows)} ({n_vis / len(rows):.1%}) — ожидаем ~50%")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    cfg = OmegaConf.load("params.yaml")
    main(cfg)
