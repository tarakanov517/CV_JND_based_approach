import numpy as np

class ImageConverter:
    def __init__(self, L_MIN: float = 0.1, L_MAX: float = 300):

        self.M = np.array([
            [0.4124564,  0.3575761,  0.1804375],
            [0.2126729,  0.7151522,  0.0721750],
            [0.0193339,  0.1191920,  0.9503041]
        ])
        self.M_inv = np.array([
            [ 3.1338561, -1.6168667, -0.4906146],
            [-0.9787684,  1.9161415,  0.0334540],
            [ 0.0719453, -0.2289914,  1.4052427]
        ])

        self.L_MAX = L_MAX
        self.L_MIN = L_MIN
        self.coord_white = np.array([0.3127, 0.3290, 0.3583])
        self.eps = 1e-10

    def sRGB_Companding(self, linear_image: np.ndarray) -> np.ndarray:
        linear = np.clip(np.asarray(linear_image, dtype=np.float64), 0.0, 1.0)
        
        encoded = np.where(
            linear <= 0.0031308,
            linear * 12.92,
            1.055 * (linear ** (1.0 / 2.4)) - 0.055
        )
        return encoded

    def Inverse_sRGB_Companding(self, image: np.ndarray) -> np.ndarray:
        img = np.asarray(image, dtype=np.float64)
        
        linear = np.where(
            img <= 0.04045,
            img / 12.92,
            ((img + 0.055) / 1.055) ** 2.4
        )
        return linear

    def Linear_RGB_to_XYZ(self, image: np.ndarray) -> np.ndarray:
        XYZ_image = image.reshape(-1, 3)
        return (XYZ_image @ self.M.T).reshape(image.shape)

    def XYZ_to_xyzL(self, XYZ_image: np.ndarray) -> np.ndarray:
        """
        Возвращает тензор (H, W, С), С = 4 (x, y, z, L)
        """
        sum_xyz = XYZ_image.sum(keepdims=True, axis=-1)

        xyz = XYZ_image / (sum_xyz + self.eps)
        
        is_black = (sum_xyz < self.eps).squeeze(-1) 

        if np.any(is_black):
            xyz[is_black] = self.coord_white

        Y = XYZ_image[..., 1]

        L_norm = np.clip(Y, 0.0, 1.0)

        L_physical = self.L_MIN + L_norm * (self.L_MAX - self.L_MIN)

        L_physical = np.expand_dims(L_physical, axis=-1)

        xyzL = np.concatenate([xyz, L_physical], axis=-1) # (H, W, С) С = 4 (x, y, z, L)

        return xyzL

    def RGB_to_xyzL(self, image: np.ndarray) -> np.ndarray:
        """
        На вход изображение - ненормализованный тензор RGB
        Возвращается тензор (x, y, z, L). L - физическая яркость на основе параметров монитора, в диапазоне (L_MIN, L_MAX)
        """
    
        # [0, 1]
        norm_img = np.asarray(image, dtype=np.float64) / 255.0
    
        # sRGB -> linRGB
        linRGBimg = self.Inverse_sRGB_Companding(norm_img)
    
        #linRGB -> XYZ
        XYZ_image = self.Linear_RGB_to_XYZ(linRGBimg)
    
        xyzL_image = self.XYZ_to_xyzL(XYZ_image)

        return xyzL_image

    def L_to_gray(self, L: np.ndarray | float) -> np.ndarray | np.uint8:
        """
        На вход L - число или матрица ненормированных яркостей
        Возвращается матрица или число (np.uint8) со значениями в градациях серого (от 0 до 255)
        """

        Y_lin = (L - self.L_MIN) / (self.L_MAX - self.L_MIN)

        Y_lin = np.clip(Y_lin, 0.0, 1.0)

        V = self.sRGB_Companding(Y_lin)

        gray_matrix = np.rint(V * 255).astype(np.uint8)

        return gray_matrix