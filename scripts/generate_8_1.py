"""
Генерация датасета эксперимента 8.1, серия [A19] (Симкин, рис. 8.11).

Варьируем:
    phi_a -- размер объекта: 1, 2, 4, 8, 16, 32, 64 пикселей
    epsilon -- относительный размах выброса: 0, 4, 16, 32
    slope_form -- форма спада: 6 видов по формулам 8.1–8.6
    psi -- целевое отношение сигнал/шум, лог-равномерно в [PSI_MIN, PSI_MAX]

Контраст НЕ сэмплируется вслепую: задаём целевое psi = ‖s‖/(d'·σ) и подбираем
L_inner так, чтобы энергия сигнала попала в psi·d'·σ. psi симметрично (в логарифме)
вокруг порога 1 → полный спектр сложности (очевидно видно / не видно / граница) и ~50/50.

Запуск:
    python scripts/generate_8_1.py
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

from src.stimulus import luminance_map_to_canvas, add_overshoots
from src.display import cd_to_pixel
from src.common import make_dir
from src.visibility_funcs import is_visible_8_1

PHI_A_VALUES = [1, 2, 4, 8, 16, 32, 64]     # размер объекта (пикс.)
EPSILON_VALUES = [0, 4, 16, 32]             # относительный размах выброса
SLOPE_FORMS = [                             # формы спада
    'sharp', 'hyperbolic', 'exponential', 'gaussian', 'triangle', 'cosine',
]
PHI_V = 8                                    # ширина выброса (пикс.), фиксирована
SAMPLES_PER_COMBO = 5                        # проб на комбинацию параметров
PSI_MIN, PSI_MAX = 0.2, 5.0                  # диапазон целевого сигнал/шум (порог = 1)
L_OUTER_MIN = 3.0                            # нижняя граница яркости фона (cd/m²)


def signal_energy(cfg: DictConfig, L_outer: float, L_inner: float,
                   phi_a: int, epsilon: float, slope: str) -> float:
    pv = PHI_V
    size = phi_a + 2 * pv
    patch = np.full((size, size), L_outer, dtype=np.float32)
    patch[pv:pv + phi_a, pv:pv + phi_a] = L_inner
    if epsilon > 0:
        add_overshoots(patch, pv, pv, phi_a, pv, epsilon,
                       abs(L_outer - L_inner), slope_form=slope)
    canvas = luminance_map_to_canvas(patch, cfg)
    field_p = float(cd_to_pixel(L_outer, cfg))
    s = canvas.astype(np.float32) - field_p
    return float(np.sqrt(np.sum(s * s)))


def solve_L_inner(cfg: DictConfig, L_outer: float, target_energy: float,
                   phi_a: int, epsilon: float, slope: str) -> float:
    Lmin = float(cfg.display.min_brightness)
    dL_max = float(L_outer) - Lmin
    if dL_max <= 0:
        return float(L_outer)
    if signal_energy(cfg, L_outer, Lmin, phi_a, epsilon, slope) <= target_energy:
        return Lmin
    lo, hi = 0.0, dL_max
    for _ in range(25):
        mid = 0.5 * (lo + hi)
        e = signal_energy(cfg, L_outer, L_outer - mid, phi_a, epsilon, slope)
        if e < target_energy:
            lo = mid
        else:
            hi = mid
    return float(np.clip(L_outer - 0.5 * (lo + hi), Lmin, L_outer))


def make_lmap_8_1(cfg: DictConfig, phi_a: int, square: str):
    """
    Карта яркостей для эксп. 8.1:
    -- оба поля 128 x 128 заполнены L_outer;
    -- квадрат phi_a x phi_a с яроксть L_inner (только в одном поле).

    Возвращает (lmap, ix, iy) — карту и координаты inner box.
    """
    s = cfg.stimulus
    L = cfg.luminance

    lmap = np.full((s.canvas_height, s.canvas_width), L.background, dtype=np.float32)

    center_x = s.canvas_width // 2
    x1 = center_x - s.margin - s.outer_box_size
    x2 = center_x + s.margin

    # Оба outer box
    for bx in (x1, x2):
        lmap[s.center_y: s.center_y + s.outer_box_size,
             bx: bx + s.outer_box_size] = L.outer

    # Inner box — только в целевом поле
    bx = x1 if square == 'left' else x2
    diff = (s.outer_box_size - phi_a) // 2
    iy = s.center_y + diff
    ix = bx + diff
    lmap[iy : iy + phi_a, ix : ix + phi_a] = L.inner

    return lmap, ix, iy


def process_one(task: dict, out_dir: Path) -> dict:
    cfg = task["cfg"]
    phi_a = task["phi_a"]
    epsilon = task["epsilon"]
    slope = task["slope"]
    seed = task["seed"]
    idx = task["idx"]

    np.random.seed(seed)
    square = 'left' if np.random.random() > 0.5 else 'right'

    lmap, ix, iy = make_lmap_8_1(cfg, phi_a, square)

    if epsilon > 0:
        delta_L = abs(float(cfg.luminance.outer) - float(cfg.luminance.inner))
        add_overshoots(lmap, ix, iy, phi_a, PHI_V, epsilon, delta_L,
                       slope_form=slope)

    canvas_clean = luminance_map_to_canvas(lmap, cfg)        # до шума — для детектора
    # canvas = add_noise(canvas_clean, sigma=cfg.noise.sigma)

    label = f"phi{phi_a:02d}_eps{epsilon:02d}_{slope}"
    filename = f"{label}_{idx:04d}.png"
    cv2.imwrite(str(out_dir / filename), cv2.cvtColor(canvas_clean, cv2.COLOR_GRAY2BGR))

    return {
        "filename": filename,
        "phi_a": phi_a,
        "epsilon": epsilon,
        "slope": slope,
        "phi_v": PHI_V,
        "psi": task["psi"],
        "L_outer": round(float(cfg.luminance.outer), 2),
        "L_inner": round(float(cfg.luminance.inner), 2),
        "square": square,
        "visible": is_visible_8_1(canvas_clean, ix, iy, phi_a, PHI_V, cfg),
    }


def build_tasks(cfg: DictConfig) -> list[dict]:
    random.seed(cfg.general.seed)
    tasks = []
    idx = 0
    d = cfg.display
    thr = float(cfg.visual.d_prime) * float(cfg.noise.sigma)   # порог энергии (ψ=1)

    for phi_a in PHI_A_VALUES:
        for epsilon in EPSILON_VALUES:
            # При epsilon=0 форма спада роли не играет -- берём только одну
            slopes = SLOPE_FORMS if epsilon > 0 else [SLOPE_FORMS[0]]
            for slope in slopes:
                for _ in range(SAMPLES_PER_COMBO):
                    # целевое ψ лог-равномерно вокруг порога → полный спектр + ~50/50
                    psi = float(np.exp(random.uniform(np.log(PSI_MIN), np.log(PSI_MAX))))
                    # яркость фона — для разнообразия адаптации; не темнее L_OUTER_MIN,
                    # иначе разницу на экране не разглядеть (тёмный край не отображается)
                    L_outer = float(np.exp(random.uniform(
                        np.log(L_OUTER_MIN), np.log(d.max_brightness))))
                    L_inner = solve_L_inner(cfg, L_outer, psi * thr, phi_a, epsilon, slope)
                    sample_cfg = OmegaConf.merge(
                        cfg, {"luminance": {"outer": L_outer, "inner": L_inner}})
                    tasks.append({
                        "cfg": sample_cfg,
                        "phi_a": phi_a,
                        "epsilon": epsilon,
                        "slope": slope,
                        "psi": round(psi, 3),
                        "seed": cfg.general.seed + idx,
                        "idx": idx,
                    })
                    idx += 1

    return tasks


def main(cfg: DictConfig):
    out_dir  = make_dir(Path(cfg.general.output_folder) / "dataset_8_1", delete_if_exist=True)
    csv_path = Path(cfg.general.output_folder) / "labels_8_1.csv"

    tasks = build_tasks(cfg)
    rows = Parallel(n_jobs=os.cpu_count(), backend="threading")(
        delayed(process_one)(task, out_dir)
        for task in tqdm(tasks)
    )

    fieldnames = ["filename", "phi_a", "epsilon", "slope", "phi_v", "psi",
                  "L_outer", "L_inner", "square", "visible"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

if __name__ == "__main__":
    cfg = OmegaConf.load("params.yaml")
    main(cfg)