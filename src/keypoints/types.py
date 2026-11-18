from dataclasses import dataclass, field
import numpy as np

@dataclass(frozen=True)
class ClusterInfo:
    """Параметры найденного объекта: центр, радиус, угловой размер и вес внимания."""
    center: tuple[float, float]
    radius_px: float
    angular_size_rad: float
    weight: float
    num_pixels: int
    points: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
