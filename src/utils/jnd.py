from dataclasses import dataclass
import math
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import norm

from src.geometry import ViewingGeometry

@dataclass(frozen=True)
class AdaptationResult:
    """Результат поиска оптимальной яркости La."""
    La: float
    loss: float
    success: bool
    iterations: int
    function_evaluations: int
    message: str

class FoveatedJND:
    """Модель JND на основе фовеального зрения и точек фиксации."""

    def __init__(
        self,
        *,
        geometry: ViewingGeometry | None = None,
        exposure_time: float = 1.0,
        angular_size_rad: float | None = None,
        detection_probability: float = 0.75,
        absolute_threshold: float = 1e-6,
        luminance_floor: float = 1e-6,
        l_min: float = 0.1,
        l_max: float = 300.0,
        l_gray: float = 60.0
    ) -> None:
        if geometry is not None:
            self.l_min = geometry.l_min
            self.l_max = geometry.l_max
            self.phi_d = float(angular_size_rad) if angular_size_rad is not None else geometry.rad_per_px()
        else:
            self.l_min = float(l_min)
            self.l_max = float(l_max)
            self.phi_d = float(angular_size_rad) if angular_size_rad is not None else 1e-3

        self.t = float(exposure_time)
        self.p = float(detection_probability)
        self.L0 = float(absolute_threshold)
        self.luminance_floor = float(luminance_floor)
        self.l_gray = float(l_gray)

        self.la = self.l_gray
        self.bounds = np.empty(0, dtype=np.float64)
        self.len_left = 0

    @staticmethod
    def _K(lambd, b1=0.98, b2=0.1, lambd1=2e-2):
        lambd = np.asarray(lambd, dtype=np.float64)
        return np.where(lambd <= 1.0, b1 * (1.0 + lambd1 / lambd), lambd**b2)

    @staticmethod
    def _A(La, a1=4.8e6, a2=7.1e3, a3=4.5e-4):
        return 1.0 + np.sqrt(a1 * La) + a2 * La * (1.0 + np.sqrt(a3 * La))

    def _Lambda(self, lambd):
        return lambd * self._K(lambd)

    @staticmethod
    def _eta(La, L3=1e6, L4=1.0, a4=600.0):
        return 1.0 + L3 / (1.0 + a4 * La * (L4 + La))

    def _C(self, La):
        tau_min = 0.05
        T = self.t / (tau_min * self._eta(La))
        return 1.0 + 1.0 / T

    @staticmethod
    def _func_phi_1(La, c2=0.25, L1=10.0, L2=2.6e-5):
        return (1.0 + L1 / (L2 + La)) ** c2

    def _S(self, La, theta_1=6.1, a_d=2.0, inf_phi_1=1.0, phi_d: float | np.ndarray | None = None):
        phi_0_min = 0.5 * (1.0 / 60.0) * (np.pi / 180.0)
        phi_0 = self._func_phi_1(La) * phi_0_min / inf_phi_1
        target_phi = np.asarray(phi_d, dtype=np.float64) if phi_d is not None else self.phi_d
        return (theta_1 * phi_0 / target_phi + 1.0) ** a_d

    def _l(self, La):
        return self.L0 * self._A(La)

    def _P(self, p_ref=0.75):
        return abs(norm.ppf(self.p) / norm.ppf(p_ref))

    def D(self, L, La, phi_d: float | np.ndarray | None = None):
        """Порог различимости яркости D(L, La)."""
        L = np.maximum(np.asarray(L, dtype=np.float64), self.luminance_floor)
        La = np.maximum(np.asarray(La, dtype=np.float64), self.luminance_floor)
        lambd = L / (1e2 * self._l(La))
        return self.L0 * self._A(La) * self._C(La) * self._Lambda(lambd) * self._S(La, phi_d=phi_d) * self._P()

    def loss(self, La: float, luminance: np.ndarray, probability: np.ndarray, phi_d: float | np.ndarray | None = None) -> float:
        """Функция потерь симметрии порогов JND."""
        if not np.isfinite(La) or La <= 0:
            return math.inf
        ratio = self.D(luminance, La, phi_d=phi_d) / self.D(La, luminance, phi_d=phi_d)
        if np.any(~np.isfinite(ratio)) or np.any(ratio <= 0):
            return math.inf
        return float(np.sum(probability * np.abs(np.log(ratio))))

    def find_la(
        self,
        fixation_luminances: list[float] | np.ndarray,
        weights: list[float] | np.ndarray | None = None,
        angular_sizes: list[float] | np.ndarray | float | None = None
    ) -> float:
        """Находит общую яркость адаптации La по точкам фиксации."""
        lums = np.asarray(fixation_luminances, dtype=np.float64).ravel()
        if weights is not None:
            w = np.asarray(weights, dtype=np.float64).ravel()
            probs = w / np.sum(w)
        else:
            probs = np.full(lums.shape, 1.0 / lums.size)

        phis = np.asarray(angular_sizes, dtype=np.float64).ravel() if angular_sizes is not None else None

        lower = max(float(np.min(lums)) / 10.0, self.luminance_floor)
        upper = float(np.max(lums)) * 10.0

        def objective(log_la):
            val = math.exp(float(log_la))
            return self.loss(val, lums, probs, phi_d=phis)

        opt = minimize_scalar(
            objective,
            bounds=(math.log(lower), math.log(upper)),
            method="bounded"
        )
        self.la = math.exp(float(opt.x))
        return self.la

    def build_level_boundaries(self) -> np.ndarray:
        """Строит границы различимых уровней яркости JND."""
        right_bounds: list[float] = []
        curr_l = self.la
        while curr_l < self.l_max:
            step = float(self.D(L=curr_l, La=self.la))
            if step <= 0:
                break
            curr_l += step
            right_bounds.append(min(curr_l, self.l_max))

        left_bounds: list[float] = []
        curr_l = self.la
        while curr_l > self.L0:
            step = float(self.D(L=curr_l, La=self.la))
            if step <= 0:
                break
            curr_l -= step
            left_bounds.append(max(curr_l, self.L0))

        self.len_left = len(left_bounds)
        left_bounds.reverse()
        self.bounds = np.array(left_bounds + [self.la] + right_bounds, dtype=np.float64)
        return self.bounds

    def l_to_k(self, l_val: float | np.ndarray) -> int | np.ndarray:
        """Переводит физическую яркость в номер уровня восприятия."""
        idx = np.searchsorted(self.bounds, l_val, side="right") - 1
        idx = np.clip(idx, 0, len(self.bounds) - 1)
        return idx - self.len_left
