import numpy as np
import cv2

from src.geometry import ViewingGeometry
from src.foveation.retina import RetinalBiology
from src.utils.converter import ImageConverter

def blur_for_keypoint(
    image: np.ndarray,
    keypoint: tuple[float, float],
    deg_per_px: float,
    retina: RetinalBiology,
    base_sigma: float = 0.2,
    scale_density: float = 150.0,
    quantization_step: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """Размывает картинку вокруг точки фиксации по плотности клеток сетчатки."""
    h, w = image.shape[:2]
    cx, cy = keypoint

    y, x = np.ogrid[:h, :w]
    dist_px = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    ecc_deg_map = dist_px * deg_per_px
    ecc_mm_map = ecc_deg_map * retina.mm_per_deg
    density_map = retina.density_interp(ecc_mm_map)

    spacing = 1.0 / np.sqrt(np.maximum(density_map, 1.0))
    sigma_map = base_sigma + scale_density * spacing

    result = np.zeros_like(image, dtype=np.float64)
    weight = np.zeros((h, w), dtype=np.float64)

    min_s = float(np.min(sigma_map))
    max_s = float(np.max(sigma_map))
    curr_s = min_s

    while curr_s <= max_s:
        next_s = curr_s + quantization_step
        mask = (sigma_map >= curr_s) & (sigma_map < next_s)

        if np.any(mask):
            ksize = int(6 * curr_s + 1) | 1
            ksize = max(3, ksize)

            blurred_patch = cv2.GaussianBlur(
                image.astype(np.float64),
                (ksize, ksize),
                sigmaX=curr_s,
                sigmaY=curr_s
            )
            result[mask] += blurred_patch[mask]
            weight[mask] += 1.0

        curr_s = next_s

    weight = np.maximum(weight, 1e-6)
    if image.ndim == 3:
        final_img = result / weight[..., None]
    else:
        final_img = result / weight

    blurred_uint8 = np.clip(final_img, 0, 255).astype(np.uint8)
    return blurred_uint8, density_map

def weighting_for_keypoint(
    image: np.ndarray,
    keypoint: tuple[float, float] | object,
    geometry: ViewingGeometry,
    retina: RetinalBiology,
    converter: ImageConverter,
    base_sigma: float = 0.2,
    scale_density: float = 150.0
) -> float:
    """Считает физическую яркость адаптации La (кд/м2) в точке взгляда."""
    kp = keypoint.center if hasattr(keypoint, "center") else keypoint
    deg_per_px = geometry.deg_per_px(image.shape[1])

    blurred_img, density_map = blur_for_keypoint(
        image=image,
        keypoint=kp,
        deg_per_px=deg_per_px,
        retina=retina,
        base_sigma=base_sigma,
        scale_density=scale_density
    )

    l_physical = converter.RGB_to_xyzL(blurred_img)[..., 3]
    total_density = np.sum(density_map)

    if total_density == 0:
        return 0.0

    return float(np.sum(l_physical * density_map) / total_density)

def extract_fixation_luminances(
    image: np.ndarray,
    keypoints: list[tuple[float, float]] | list[object],
    geometry: ViewingGeometry,
    retina: RetinalBiology,
    converter: ImageConverter,
    base_sigma: float = 0.2,
    scale_density: float = 150.0
) -> list[float]:
    """Считает яркость La для всех переданных точек взгляда."""
    if not keypoints:
        return []

    return [
        weighting_for_keypoint(
            image=image,
            keypoint=kp,
            geometry=geometry,
            retina=retina,
            converter=converter,
            base_sigma=base_sigma,
            scale_density=scale_density
        )
        for kp in keypoints
    ]
