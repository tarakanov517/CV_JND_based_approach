import numpy as np
import rootutils
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator="src", pythonpath=True)


def pixel_to_cd(p: float, cfg: DictConfig) -> float:
    """Пиксельное значение (0–255) -> яркость в cd/m^2."""
    d = cfg.display
    return d.min_brightness + (d.max_brightness - d.min_brightness) * (p / 255) ** d.gamma


def cd_to_pixel(L: float, cfg: DictConfig) -> int:
    """Яркость в cd/m^2 -> пиксельное значение (0–255)."""
    d = cfg.display
    ratio = (L - d.min_brightness) / (d.max_brightness - d.min_brightness)
    ratio = np.clip(ratio, 0.0, 1.0)
    return np.round(ratio ** (1.0 / d.gamma) * 255).astype(np.uint8)


def phi0_arcmin(cfg: DictConfig) -> float:
    """Угловой размер одного пикселя в угловых минутах."""
    pixel_mm = 25.4 / cfg.display.ppi
    return np.degrees(np.arctan(pixel_mm / (cfg.display.distance_cm * 10))) * 60
