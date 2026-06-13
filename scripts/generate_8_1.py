"""
Генерация датасета эксперимента 8.1, серия [A19] (Симкин, рис. 8.11).

Варьируем:
    phi_a -- размер объекта: 1, 2, 4, 8, 16, 32, 64 пикселей
    epsilon -- относительный размах выброса: 0, 4, 16, 32
    slope_form -- форма спада: 6 видов по формулам 8.1–8.6
    delta_L -- контраст (сэмплируется случайно)

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

from src.stimulus import luminance_map_to_canvas, add_overshoots, add_noise
from src.common import make_dir
from src.visibility_funcs import is_visible_8_1

PHI_A_VALUES = [1, 2, 4, 8, 16, 32, 64]     # размер объекта (пикс.)
EPSILON_VALUES = [0, 4, 16, 32]             # относительный размах выброса
SLOPE_FORMS = [                             # формы спада
    'sharp', 'hyperbolic', 'exponential', 'gaussian', 'triangle', 'cosine',
]
PHI_V = 8                                    # ширина выброса (пикс.), фиксирована
SAMPLES_PER_COMBO = 5                        # проб на комбинацию параметров


def sample_luminance(cfg: DictConfig) -> DictConfig:
    """Случайно выбирает L_outer и L_inner (темнее на фоне L_outer) в динамическом диапазоне дисплея."""
    d = cfg.display
    L_outer = float(np.exp(random.uniform(
        np.log(d.min_brightness + 0.01), np.log(d.max_brightness)
    )))
    dL = random.uniform(0.01, 1.5) * L_outer
    L_inner = float(np.clip(L_outer - dL, d.min_brightness, d.max_brightness))
    return OmegaConf.merge(cfg, {"luminance": {"outer": L_outer, "inner": L_inner}})


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
    canvas = add_noise(canvas_clean, sigma=cfg.noise.sigma)

    label = f"phi{phi_a:02d}_eps{epsilon:02d}_{slope}"
    filename = f"{label}_{idx:04d}.png"
    cv2.imwrite(str(out_dir / filename), cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR))

    return {
        "filename": filename,
        "phi_a": phi_a,
        "epsilon": epsilon,
        "slope": slope,
        "phi_v": PHI_V,
        "L_outer": round(float(cfg.luminance.outer), 2),
        "L_inner": round(float(cfg.luminance.inner), 2),
        "square": square,
        "visible": is_visible_8_1(canvas_clean, ix, iy, phi_a, PHI_V, cfg),
    }


def build_tasks(cfg: DictConfig) -> list[dict]:
    random.seed(cfg.general.seed)
    tasks = []
    idx = 0

    for phi_a in PHI_A_VALUES:
        for epsilon in EPSILON_VALUES:
            # Общая яркость для всех форм спада при данных (phi_a, epsilon),
            # чтобы сравнение форм не путалось со случайным разбросом яркостей.
            sample_cfg = sample_luminance(cfg)
            # При epsilon=0 форма спада роли не играет -- берём только одну
            slopes = SLOPE_FORMS if epsilon > 0 else [SLOPE_FORMS[0]]
            for slope in slopes:
                for _ in range(SAMPLES_PER_COMBO):
                    tasks.append({
                        "cfg": sample_cfg,
                        "phi_a": phi_a,
                        "epsilon": epsilon,
                        "slope": slope,
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

    fieldnames = ["filename", "phi_a", "epsilon", "slope", "phi_v",
                  "L_outer", "L_inner", "square", "visible"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

if __name__ == "__main__":
    cfg = OmegaConf.load("params.yaml")
    main(cfg)
