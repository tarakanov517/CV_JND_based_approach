from pathlib import Path
from dataclasses import dataclass
import yaml
import numpy as np
from scipy.interpolate import interp1d

from src.paths import CONFIGS_DIR

@dataclass(frozen=True)
class RetinalBiology:
    """Параметры сетчатки: масштаб мм/град и интерполяция плотности клеток (RGC)."""

    mm_per_deg: float
    density_interp: interp1d

    def get_density(self, ecc_mm: np.ndarray | float) -> np.ndarray | float:
        """Плотность клеток сетчатки по расстоянию от центра в мм."""
        return self.density_interp(ecc_mm)

def load_retina_biology(config_path: Path | str | None = None) -> RetinalBiology:
    """Загружает параметры сетчатки из foveation_config.yaml."""
    if config_path is None:
        config_path = CONFIGS_DIR / "foveation_config.yaml"

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)["retina"]

    ecc = np.array(data["rgc_density"]["eccentricity_mm"], dtype=np.float64)
    dens = np.array(data["rgc_density"]["density"], dtype=np.float64)

    interp = interp1d(
        ecc, dens,
        kind="linear",
        bounds_error=False,
        fill_value=(dens[0], dens[-1])
    )

    return RetinalBiology(
        mm_per_deg=float(data["mm_per_deg"]),
        density_interp=interp
    )
