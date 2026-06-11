import numpy as np
import rootutils
from omegaconf import DictConfig
from scipy.ndimage import distance_transform_edt
from scipy.special import erf

rootutils.setup_root(__file__, indicator="src", pythonpath=True)

from src.display import cd_to_pixel, pixel_to_cd


def luminance_map_to_canvas(lmap: np.ndarray, cfg: DictConfig) -> np.ndarray:
    """Переводит массив яркостей (cd/m^2) в пиксельный массив uint8."""
    d = cfg.display
    ratio = (lmap - d.min_brightness) / (d.max_brightness - d.min_brightness)
    ratio = np.clip(ratio, 0.0, 1.0)
    return np.round(ratio ** (1.0 / d.gamma) * 255).astype(np.uint8)


def make_luminance_map(cfg: DictConfig) -> np.ndarray:
    """
    Промежуточный шаг: массив яркостей в cd/m^2, каждый элемент = один пиксель.
    Возвращает float32-массив размером (canvas_height, canvas_width).
    """
    s = cfg.stimulus
    L = cfg.luminance

    lmap = np.full((s.canvas_height, s.canvas_width), L.background, dtype=np.float32)

    center_x = s.canvas_width // 2
    x1 = center_x - s.margin - s.outer_box_size
    x2 = center_x + s.margin

    for x in (x1, x2):
        lmap[s.center_y : s.center_y + s.outer_box_size,
             x : x + s.outer_box_size] = L.outer
        lmap[s.center_y + s.diff : s.center_y + s.diff + s.inner_box_size,
             x + s.diff : x + s.diff + s.inner_box_size] = L.inner

    return lmap


def make_canvas(cfg: DictConfig) -> np.ndarray:
    """
    Создаёт базовый canvas с двумя полями (левым и правым).
    Возвращает grayscale-массив uint8.
    """
    return cd_to_pixel(make_luminance_map(cfg), cfg)


def _corner_coords(row: int, col: int, corner: str, ix: int, iy: int,
                   inner_box_size: int):
    if corner == 'tl':
        return iy + row, ix + col
    elif corner == 'tr':
        return iy + row, ix + inner_box_size - 1 - col
    elif corner == 'bl':
        return iy + inner_box_size - 1 - row, ix + col
    elif corner == 'br':
        return iy + inner_box_size - 1 - row, ix + inner_box_size - 1 - col


def cut_corner(canvas: np.ndarray, ix: int, iy: int, corner: str,
               theta: int, style: str, fill: int,
               inner_box_size: int) -> None:
    """
    Срезает угол внутреннего квадрата.

    style:
        'triangle' -- треугольный срез (гипотенуза прямоугольного, равнобедренного треугольника)
        'rect' -- прямоугольный срез (весь блок h×w)
        'circle' -- круговой срез (четверть окружности)
    """
    h = w = theta
    
    for row in range(h):
        for col in range(w):
            if style == 'triangle' and row / h + col / w >= 1:
                continue
            if style == 'circle' and (row - h) ** 2 + (col - w) ** 2  <= h ** 2:
                continue
            py, px = _corner_coords(row, col, corner, ix, iy, inner_box_size)
            canvas[py, px] = fill


def cut_niche(canvas: np.ndarray, ix: int, iy: int, side: str,
              niche_h: int, niche_w: int, fill: int,
              inner_box_size: int) -> None:
    """
    Прямоугольная ниша по центру одной из сторон.

    niche_h: глубина ниши (внутрь квадрата)
    niche_w: ширина ниши вдоль стороны
    """
    offset = (inner_box_size - niche_w) // 2

    if side == 'top':
        r0, r1 = iy, iy + niche_h
        c0, c1 = ix + offset, ix + offset + niche_w
    elif side == 'bottom':
        r0, r1 = iy + inner_box_size - niche_h,  iy + inner_box_size
        c0, c1 = ix + offset, ix + offset + niche_w
    elif side == 'left':
        r0, r1 = iy + offset, iy + offset + niche_w
        c0, c1 = ix, ix + niche_h
    elif side == 'right':
        r0, r1 = iy + offset, iy + offset + niche_w
        c0, c1 = ix + inner_box_size - niche_h, ix + inner_box_size

    canvas[r0:r1, c0:c1] = fill


def deformation(canvas: np.ndarray, cfg: DictConfig,
                mode: str, params: tuple,
                square: str = 'right') -> None:
    """
    Применяет деформацию к одному из квадратов.

    mode / params:
        'triangle'  (theta) -- треугольные срезы всех 4 углов
        'rect'      (theta) -- прямоугольные срезы всех 4 углов
        'circle'    (theta) -- круговые срезы всех 4 углов
        'niche'     (side, h, w) -- ниша на одной стороне
    """
    s = cfg.stimulus
    center_x = s.canvas_width // 2
    x1 = center_x - s.margin - s.outer_box_size
    x2 = center_x + s.margin

    ix = (x1 if square == 'left' else x2) + s.diff
    iy = s.center_y + s.diff
    fill = cd_to_pixel(cfg.luminance.outer, cfg)

    if mode in ('triangle', 'rect', 'circle'):
        theta = params
        for corner in ('tl', 'tr', 'bl', 'br'):
            cut_corner(canvas, ix, iy, corner, theta,
                       style=mode, fill=fill,
                       inner_box_size=s.inner_box_size)

    elif mode == 'niche':
        side, niche_h, niche_w = params
        cut_niche(canvas, ix, iy, side, niche_h, niche_w,
                  fill=fill, inner_box_size=s.inner_box_size)


def add_noise(canvas: np.ndarray, sigma: float) -> np.ndarray:
    """Добавляет аддитивный гауссов шум. Новая реализация при каждом вызове."""
    noise = np.random.normal(0, sigma, canvas.shape).astype(np.float32)
    return np.clip(canvas.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def overshoot_profile(dist: np.ndarray, phi_v: float, epsilon: float,
                       slope_form: str,
                       beta: float = 2.0,
                       a_gauss: float = 0.1) -> np.ndarray:
    """
    Нормированный профиль выброса как функция расстояния от границы (пикс.).

    Формулы 8.1-8.6 из диссертации Симкина:
        'sharp'       -- резкий (одноэлементный)
        'hyperbolic'  -- усечённый гиперболический
        'exponential' -- экспоненциальный (параметр beta)
        'gaussian'    -- гауссов интеграл ошибок (параметр a_gauss)
        'triangle'    -- треугольный
        'cosine'      -- косинусоидальный

    Возвращает массив float32, значения в [0, epsilon].
    """
    tmax = float(phi_v)
    amp  = np.zeros_like(dist, dtype=np.float32)

    if slope_form == 'sharp':                                        # (8.1)
        # Одноэлементный выброс: все 8-связные соседи границы.
        amp = np.where(dist <= np.sqrt(2), epsilon, 0.0).astype(np.float32)

    elif slope_form == 'hyperbolic':                                 # (8.2)
        # При tmax опорное значение, если меньше - переходим к резкому выбросу
        if tmax > 1.0:
            hyp = epsilon * (tmax - dist) / (dist * (tmax - 1.0))
            amp = np.where(dist < tmax, hyp, 0.0).astype(np.float32)
        else:
            amp = np.where(dist <= 1.0, epsilon, 0.0).astype(np.float32)

    elif slope_form == 'exponential':                                # (8.3)
        # При dist=1 у нас будет плато eps, потом резкий спад
        amp = (epsilon * np.exp((1.0 - dist) / beta)).astype(np.float32)

    elif slope_form == 'gaussian':                                   # (8.4)
        amp = (epsilon * (1.0 - erf(a_gauss * dist / np.sqrt(2.0)))).astype(np.float32)

    elif slope_form == 'triangle':                                   # (8.5)
        amp = np.where(dist < tmax,
                       epsilon * (1.0 - dist / tmax),
                       0.0).astype(np.float32)

    elif slope_form == 'cosine':                                     # (8.6)
        amp = np.where(dist < tmax,
                       epsilon * np.cos(np.pi / 2.0 * dist / tmax),
                       0.0).astype(np.float32)

    return amp


def add_overshoots(lmap: np.ndarray,
                   ix: int, iy: int, inner_size: int,
                   phi_v: int, epsilon: float, delta_L: float,
                   slope_form: str = 'cosine',
                   **slope_kwargs) -> np.ndarray:
    """
    Добавляет униполярный яркостный выброс снаружи периметра inner box.

    Выброс добавляется в карту яркостей (float32) до конвертации в uint8,
    в кольцо шириной phi_v пикселей вокруг квадрата ix,iy -> inner_size x inner_size.

    Параметры:
        lmap       : float32-карта яркостей (изменяется на месте)
        ix, iy     : левый верхний угол inner box
        inner_size : размер inner box в пикселях
        phi_v      : ширина выброса (пикс.)
        epsilon    : относительный размах ΔLв / ΔLа
        delta_L    : ΔLа = |L_outer - L_inner| (cd/m^2)
        slope_form : форма спада (см. _overshoot_profile)

    Возвращает lmap.
    """
    H, W = lmap.shape

    # Маска пикселей внутри inner box
    mask = np.ones((H, W), dtype=bool)
    mask[iy:iy + inner_size, ix:ix + inner_size] = False

    # Евклидово расстояние от каждого внешнего пикселя до ближайшего пикселя inner box.
    dist_map = distance_transform_edt(mask)
    
    # Область выброса: снаружи inner box и в пределах phi_v пикселей
    region = mask & (dist_map <= phi_v)

    profile = overshoot_profile(dist_map[region], phi_v, epsilon,
                                 slope_form, **slope_kwargs)
    lmap[region] += profile * delta_L
    return lmap
