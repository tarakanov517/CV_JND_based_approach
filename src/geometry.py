from dataclasses import dataclass, field
import math

@dataclass
class ViewingGeometry:
    """Параметры экрана и дистанция просмотра. Считает размер пикселя в мм и угловые размеры."""

    view_dist_mm: float = 600.0
    l_min: float = 0.1
    l_max: float = 300.0
    width_px: int = 1920
    height_px: int = 1080
    diagonal_inch: float = 16.0

    pixel_size_mm: float = field(init=False)
    width_mm: float = field(init=False)
    height_mm: float = field(init=False)
    ppi: float = field(init=False)

    def __post_init__(self):
        diag_px = math.sqrt(self.width_px ** 2 + self.height_px ** 2)
        self.ppi = diag_px / self.diagonal_inch
        self.pixel_size_mm = 25.4 / self.ppi
        self.width_mm = self.width_px * self.pixel_size_mm
        self.height_mm = self.height_px * self.pixel_size_mm

    def deg_per_px(self, img_width_px: int | None = None) -> float:
        """Градусы зрительного угла на один пиксель."""
        if img_width_px is None or img_width_px == self.width_px:
            rad = self.pixel_size_mm / self.view_dist_mm
            return rad * (180.0 / math.pi)

        scale = self.width_px / img_width_px
        rad = (self.pixel_size_mm * scale) / self.view_dist_mm
        return rad * (180.0 / math.pi)

    def rad_per_px(self, img_width_px: int | None = None) -> float:
        """Угловой размер одного пикселя в радианах."""
        deg = self.deg_per_px(img_width_px)
        return deg * (math.pi / 180.0)
