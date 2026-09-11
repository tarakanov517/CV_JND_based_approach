import math
import numpy as np
from scipy.optimize import minimize_scalar, minimize

class SimkinJNDModel:

    def __init__(self, L_MIN = 0.1, L_MAX = 300, screen_diag_inch = 16, screen_width_px = 1920, 
                 screen_height_px = 1080, distance_m = 0.5, l_gray = 60):
        
        self.L_MIN = L_MIN
        self.L_MAX = L_MAX
        self.L0 = 1e-6

        diag_px = math.sqrt(screen_width_px ** 2 + screen_height_px ** 2)
        ppi = diag_px / screen_diag_inch
        
        pixel_size_m = 0.0254 / ppi # 1 дюйм = 0.0254 метра

        self.phi_screen = pixel_size_m / distance_m # считаем угловой размер пикселя на экране
        self.phi = self.phi_screen
        
        self.screen_width_px = screen_width_px
        self.screen_height_px = screen_height_px
        self.ppi = ppi
        self.l_gray = l_gray

    def _Simkin_A(self, La: float | np.ndarray) -> float | np.ndarray:
        a1: float = 4800000.0
        a2: float = 7100.0
        a3: float = 0.00045
        return 1.0 + np.sqrt(a1 * La) + a2 * La * (1.0 + np.sqrt(a3 * La))

    def _Simkin_l(self, La: float | np.ndarray) -> float | np.ndarray:
        L0: float = 0.000001
        return L0 * self._Simkin_A(La)

    def _Simkin_lambda(self, L: np.ndarray | float, La: float | np.ndarray) -> np.ndarray | float:
        Xao: float = 100.0
        return L / (Xao * self._Simkin_l(La))

    def _Simkin_LAt(self, lambda_: np.ndarray | float) -> np.ndarray | float:
        b1: float = 0.98
        b2: float = 0.1
        lambda1: float = 0.02
        lambda_arr = np.asarray(lambda_, dtype=np.float64)
        return np.where(lambda_arr <= 1.0, b1 * (lambda_arr + lambda1), lambda_arr ** (1.0 + b2))

    def _Simkin_ro(self, La: float | np.ndarray) -> float | np.ndarray:
        L1: float = 10.0
        L2: float = 0.000026
        c2: float = 0.25
        return (1.0 + L1 / (L2 + La)) ** c2

    def _Simkin_teta(self, phi: float, La: float | np.ndarray) -> float | np.ndarray:
        phi_omin: float = (1.0 / 60.0) * np.pi / 180.0
        return phi / (phi_omin * self._Simkin_ro(La))

    def _Simkin_S(self, teta: float) -> float:
        teta1: float = 6.1
        return (teta1 / teta + 1.0) ** 2

    def _simkin_nu(self, La: float | np.ndarray) -> float | np.ndarray:
        nu = 4.1 - 3.4 / (1.0 + np.exp(-0.5 - np.log10(La)))
        return np.clip(nu, 1.0, 4.0)

    def _simkin_T(self, t: float, La: float | np.ndarray) -> float | np.ndarray:
        return t / (0.05 * self._simkin_nu(La))

    def _simkin_C(self, T: float) -> float:
        return 1.0 + 1.0 / T

    def _Simkin_fN(self, La: float | np.ndarray, L: np.ndarray | float, LN: float) -> np.ndarray | float:
        sigma_N = 3.0 * LN / self._Simkin_l(La)
        lat_val = self._Simkin_LAt(self._Simkin_lambda(L, La))
        return np.sqrt(1.0 + (sigma_N * sigma_N) / (lat_val * lat_val))

    def _simkin_core(
        self,
        La: float | np.ndarray,
        L: np.ndarray | float,
        phi: float | None = None,
        LN: float = 0.0,
        t: float = 1
    ) -> np.ndarray | float:
        
        if phi is None:
            phi = self.phi

        return (
            self.L0
            * self._Simkin_A(La)
            * self._Simkin_LAt(self._Simkin_lambda(L, La))
            * self._Simkin_S(self._Simkin_teta(phi, La))
            * self._Simkin_fN(La, L, LN)
            * self._simkin_C(self._simkin_T(t, La))
        )
    
    def find_La_with_background(self, L_patch, optimizer = 'L-BFGS-B', target_width_cm = None):
        '''
        L_patch подается в оригинальном разрешении
        
        '''
        if (isinstance(L_patch, np.ndarray) and L_patch.size > 0 and L_patch.ndim == 2):
            h_patch_px, w_patch_px = L_patch.shape
            aspect_ratio = h_patch_px / w_patch_px

            if target_width_cm is None:
                target_w_px = w_patch_px
                target_h_px = h_patch_px
            else:
                target_w_px = int(np.round((target_width_cm / 2.54) * self.ppi))
                target_h_px = int(np.round(target_w_px * aspect_ratio))

            n_patch_screen = target_w_px * target_h_px

            scale = target_w_px / w_patch_px
            self.phi = scale * self.phi_screen
        else:
            n_patch_screen = 0
            self.phi = self.phi_screen

        n_total = self.screen_width_px * self.screen_height_px
        
        n_bg = n_total - n_patch_screen

        if L_patch.size > 0:
            logL = np.log(np.maximum(L_patch, 1e-12))
            log_La_min = float(min(logL.min(), math.log(self.l_gray)))
            log_La_max = float(max(logL.max(), math.log(self.l_gray)))
        else:
            log_La_min = -3.0
            log_La_max = 6.0

        den_patch = self._simkin_core(La=L_patch, L=L_patch)
        den_bg = self._simkin_core(La=self.l_gray, L=self.l_gray)

        def objective_log_La(log_La):
            curr_La = math.exp(float(log_La))
        
            if L_patch.size > 0:
                num_patch = self._simkin_core(La=curr_La, L=L_patch)
                # sum_patch = (n_patch_screen / L_patch.size) * float(np.sum(np.abs(np.log(num_patch / den_patch)))) # версия с модулем
                sum_patch = (n_patch_screen / L_patch.size) * float(np.sum((np.log(num_patch / den_patch))**2)) # версия с квадратом
            else:
                sum_patch = 0.0
            
            num_bg = self._simkin_core(La=curr_La, L=self.l_gray)
            # sum_bg = n_bg * np.abs(math.log(num_bg / den_bg)) # версия с модулем
            sum_bg = n_bg * (math.log(num_bg / den_bg))**2 # версия с квадратом
            
            # return float((sum_patch + sum_bg) / n_total) + 1e-5 * (float(log_La) - math.log(self.l_gray))**2
            return float((sum_patch + sum_bg) / n_total)

        if optimizer == 'minimize_scalar':

            history_x = []
            history_loss = []

            def tracked_objective(log_la):
                loss = objective_log_La(log_la)
                history_x.append(float(log_la))
                history_loss.append(float(loss))
                return loss

            result = minimize_scalar(
                tracked_objective,
                bounds=(log_La_min, log_La_max),
                method='bounded',
                options={'xatol': 1e-6},
            )

            self.La = math.exp(float(result.x))

        elif optimizer == 'log_grid_search':

            log_space = np.linspace(log_La_min, log_La_max, num=1000)

            best_la = 0
            min_loss = 1e10

            for log_la in log_space:
                loss = objective_log_La(log_la)
                if min_loss > loss:
                    min_loss = loss
                    best_la = np.exp(log_la)

            self.La = best_la

        if optimizer == 'L-BFGS-B':
            
            if L_patch.size > 0:
                mean_L = (float(np.mean(L_patch)) * n_patch_screen + self.l_gray * n_bg) / n_total
            else:
                mean_L = self.l_gray
                
            mean_log = math.log(max(mean_L, 1e-12))
            
            def wrapped_objective(x):
                return objective_log_La(x.item())
            
            result = minimize(
                fun=wrapped_objective,
                x0=np.array([mean_log]),
                bounds=[(log_La_min, log_La_max)],
                method='L-BFGS-B',
                options={'ftol': 1e-9}
            )
            
            self.La = math.exp(float(result.x[0]))
        
        return self.La

    def build_level_boundaries(self):
        L_right_bounds = []
        L_curr = self.La
        while L_curr < self.L_MAX:
            L_curr += self._simkin_core(L = L_curr, La = self.La)
            L_right_bounds.append(min(L_curr, self.L_MAX))

        L_left_bounds = []
        L_curr = self.La
        while L_curr > self.L0:
            L_curr -= self._simkin_core(L = L_curr, La = self.La)
            L_left_bounds.append(max(L_curr, self.L0))

        self.len_left = len(L_left_bounds)

        L_left_bounds.reverse()

        self.bounds = np.array(L_left_bounds + [self.La] + L_right_bounds)

        return self.bounds

    def L_to_k(self, L): # всегда берем нижнюю границу интервала, в который попали
        idx = np.searchsorted(self.bounds, L, side='right') - 1
        idx = np.clip(idx, 0, len(self.bounds) - 1)
        return idx - self.len_left
        
    def k_to_L(self, k): # всегда берем нижнюю границу интервала, в который попали
        return self.bounds[k + self.len_left]