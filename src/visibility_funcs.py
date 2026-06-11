import numpy as np
import rootutils
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator="src", pythonpath=True)

from src.display import cd_to_pixel


def noise_sigma_cd(cfg: DictConfig) -> float:
    """СКЗ аддитивного шума L_N, пересчитанное из пикселей в cd/m^2
    по локальному наклону характеристики дисплея у яркости фона объекта."""
    d = cfg.display
    p = cd_to_pixel(cfg.luminance.outer, cfg)
    dL_dp = (d.max_brightness - d.min_brightness) * d.gamma * (p / 255) ** (d.gamma - 1) / 255
    return float(cfg.noise.sigma * dL_dp)


def is_visible_7_3(mode, params, cfg):
    # Обнаружение деформации в шуме с интегрированием по её площади (п. 7.3.2):
    #   ниша:  ΔL_n = d' * L_N / sqrt(θa·θb)   -- ур. (7.4)
    #   углы:  ΔL_n = d' * L_N / θ_угл         -- ур. (7.5)
    L_N = noise_sigma_cd(cfg)
    d = cfg.visual.d_prime
    dL = abs(cfg.luminance.inner - cfg.luminance.outer)

    if mode == 'niche':
        _, h, w = params
        threshold = d * L_N / np.sqrt(h * w)
    else:
        theta = params
        threshold = d * L_N / theta

    return int(dL > threshold)


def is_visible_8_1(canvas_clean: np.ndarray, ix: int, iy: int,
                   phi_a: int, phi_v: int, cfg) -> int:
    """Детектор для серии [A19], рис. 8.11.

    Та же модель, что в is_visible_7_3, обобщённая на сигнал с яркостным выбросом.
    Эмпирическое уравнение 7.4 (ΔL_n прямопропорц 1/sqrt(площади))
    объект виден, если энергия сигнала превышает порог по шуму. Здесь сигнал = объект + выброс:

        ||s|| = sqrt( sum(p_сигнал - p_поля)² )  >  d' · σ_px

    Считаем по чистому (до шума) рендеру в пиксельном домене — единицы те же, что
    y шума (σ_px добавляется в пикселях). Связь с 7.3:
      • при ε=0 кольца нет, порог ΔL = d'·L_N/sqrt(A) — ровно ур. 7.4;
      • при ε>0 кольцо добавляет энергию sum(profile^2), поэтому форма спада влияет:
        широкий спад (косинус/треугольник) набирает больше энергии, а значит ниже порог,
        что и даёт качественную картину рис. 8.11.

    Параметры:
        canvas_clean : рендер до наложения шума
        ix, iy       : левый верхний угол объекта
        phi_a        : размер объекта (пикс.)
        phi_v        : ширина кольца выброса (пикс.)
    """
    field_p = float(cd_to_pixel(cfg.luminance.outer, cfg))

    y0, x0 = max(iy - phi_v, 0), max(ix - phi_v, 0)
    y1, x1 = iy + phi_a + phi_v, ix + phi_a + phi_v

    s = canvas_clean[y0:y1, x0:x1].astype(np.float32) - field_p
    energy = float(np.sqrt(np.sum(s * s)))

    return int(energy > cfg.visual.d_prime * cfg.noise.sigma)