from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import glob
import torch
from PIL import Image
import torchvision.transforms.functional as TF
import numpy as np
from scipy.stats import norm
import cv2
import math
from scipy.optimize import minimize_scalar
import torch.nn as nn

class JNDModel:
    
    def __init__(self, L_min, L_max, t = 1, phi_d = 1e-3, p = 0.75):
        self.L_min = L_min
        self.L_max = L_max
        self.t = t
        self.phi_d = phi_d
        self.p = p
        self.L0 = 1e-6

    def _K(self, lambd, b1 = 0.98, b2 = 0.1, lambd1 = 2e-2):
        lambd = np.asarray(lambd, dtype = np.float64)
        return np.where(lambd <= 1, b1 * (1 + lambd1 / lambd), lambd ** b2)
    
    def _A(self, La, a1 = 4.8e6, a2 = 7.1e3, a3 = 4.5e-4): 
        return 1 + (a1 * La) ** 0.5 + a2 * La * (1 + (a3 * La) ** 0.5)
    
    def _Lambda(self, lambd):
        return lambd * self._K(lambd)
    
    def _eta(self, La, L3 = 1e6, L4 = 1, a4 = 600):
        return 1 + L3 / (1 + a4 * La * (L4 + La))
    
    def _C(self, La, t):
        tao_min = 0.05
        T = t / (tao_min * self._eta(La))
        return 1 + 1 / T 
    
    def _func_phi_1(self, La, c2 = 0.25, L1 = 10, L2 = 2.6e-5):
        return (1 + L1 / (L2 + La)) ** c2
    
    def _S(self, La, theta_1 = 6.1, a_d = 2, inf_phi_1 = 1):
        phi_0_min = 0.5 * (1 / 60) * (np.pi / 180)
        phi_1 = self._func_phi_1(La)
        phi_0 = phi_1 * phi_0_min / inf_phi_1
        
        return (theta_1 * phi_0 / self.phi_d + 1.0) ** a_d
    
    def _l(self, La):
        return self.L0 * self._A(La)
    
    def _P(self, p, p_ref = 0.75):
        return abs(norm.ppf(p) / norm.ppf(p_ref))

    def _D(self, L, La):
        lambd = L / (10 ** 2 * self._l(La))
        return self.L0 * self._A(La) * self._C(La, self.t) * self._Lambda(lambd) * self._S(La, self.phi_d) * self._P(self.p)

    def find_La(self, L_matrix):
        start_La = np.mean(L_matrix)
        EPS = 1e-12
        logL = np.log(np.maximum(L_matrix, EPS))
    
        # Границы можно расширить, если адаптация может лежать вне диапазона яркостей картинки.
        log_La_min = float(logL.min())
        log_La_max = float(logL.max())
    
        def S_score(L_matrix, start_La):
            numerator = self._D(L_matrix, start_La)
            denominator = self._D(L_matrix, L_matrix)
            return float(np.mean(np.log(numerator / denominator)))
        
        def objective_log_La(log_La):
            return S_score(L_matrix, math.exp(float(log_La)))
        
        result = minimize_scalar(
            objective_log_La,
            bounds=(log_La_min, log_La_max),
            method='bounded',
            options={'xatol': 1e-6},
        )
        
        log_La_star = float(result.x)
        self.La = math.exp(log_La_star)
        
        return self.La

    def build_level_boundaries(self):
        L_right_bounds = []
        L_curr = self.La
        while L_curr < self.L_max:
            L_curr += self._D(L_curr, self.La)
            L_right_bounds.append(min(L_curr, self.L_max))

        L_left_bounds = []
        L_curr = self.La
        while L_curr > self.L0:
            L_curr -= self._D(L_curr, self.La)
            L_left_bounds.append(max(L_curr, self.L0))

        self.len_left = len(L_left_bounds)

        L_left_bounds.reverse()

        self.bounds = np.array(L_left_bounds + [self.La] + L_right_bounds)

        return self.bounds
    
    def L_to_k(self, L):
        return np.searchsorted(self.bounds, L, side = 'right') - 1 - self.len_left
    
    def k_to_L(self, k):
        return self.bounds[k + self.len_left]

class ImageConverter:
    def __init__(self):
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

    def read_img(self, image):
        V = cv2.imread(image)
        V = cv2.cvtColor(V, cv2.COLOR_BGR2RGB)
        return V.astype(np.float64)

    def normalize_image(self, image):
        if image.max() > 1:
            image = image / 255.0
        return image

    def Inverse_sRGB_Companding(self, image):
        mask = image <= 0.04045
        image[mask] = image[mask] / 12.92
        image[~mask] = ((image[~mask] + 0.055) / 1.055) ** 2.4
        return image

    def Linear_RGB_to_XYZ(self, image):
        XYZ_image = image.reshape(-1, 3)
        return (XYZ_image @ self.M.T).reshape(image.shape)

    def XYZ_to_xyY(self, XYZ_image, L_max=617.0, coord_white=(0.3127, 0.3290)):
        sum_xyz = XYZ_image.sum(axis=-1, keepdims=True)
        eps = 1e-10
        xy = XYZ_image[..., :2] / (sum_xyz + eps)
        is_black = (sum_xyz.squeeze(-1) < eps)
        if np.any(is_black):
            xy[is_black] = coord_white
        Y = XYZ_image[..., 1]
        L_norm = np.clip(Y / L_max, 0, 1)
        return np.stack([xy[..., 0], xy[..., 1], L_norm], axis=-1)

    def XYZ_to_xzL(self, XYZ_image, L_max=617.0, coord_white=(0.3127, 0.3583)):
        sum_xyz = XYZ_image.sum(axis=-1)
        eps = 1e-10
        x = XYZ_image[..., 0] / (sum_xyz + eps)
        z = XYZ_image[..., 2] / (sum_xyz + eps)
        is_black = (sum_xyz < eps)
        if np.any(is_black):
            x[is_black] = coord_white[0]
            z[is_black] = coord_white[1]
        Y = XYZ_image[..., 1]
        L_norm = np.clip(Y / L_max, 0, 1)
        return np.stack([x, z, L_norm], axis=-1)

class Cifar10Dataset(Dataset):
    def __init__(self, data_dir, is_jnd=False, layers=None):
        self.layers = layers
        self.is_jnd = is_jnd
        self.data_dir = Path(data_dir)
        
        CIFAR10_CLASSES = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]

        self.classes = CIFAR10_CLASSES
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        self.samples = []
        extension = "*.npy" if is_jnd else "*.png"

        
        for cls_name in self.classes:
            cls_folder = self.data_dir / cls_name
            class_idx = self.class_to_idx[cls_name]
            
            file_paths = glob.glob(str(cls_folder / extension))
            if not is_jnd and not file_paths:
                file_paths = glob.glob(str(cls_folder / "*.jpg"))
                
            for path in file_paths:
                self.samples.append((path, class_idx))

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        
        if self.is_jnd:
            jnd_matrix = np.load(file_path).astype(np.float32)
            image = torch.from_numpy(jnd_matrix).permute(2, 0, 1)

        else:
            image = Image.open(file_path).convert('RGB')
            image = TF.to_tensor(image)

        return image, torch.tensor(label, dtype=torch.long)

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = downsample
    
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.LeakyReLU(0.1, inplace=True)

        self.layer1 = self._make_layer(64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(256, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(512, num_blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = []
        layers.append(BasicBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x