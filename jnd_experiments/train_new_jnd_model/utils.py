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
import random
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import os
import torchvision.transforms as transforms

class SimkinJNDModel:

    def __init__(self, L_MIN = 0.1, L_MAX = 300, screen_diag_inch = 16, screen_width_px = 1920, 
                 screen_height_px = 1080, distance_m = 0.5, l_gray = 60):
        
        self.L_MIN = L_MIN
        self.L_MAX = L_MAX
        self.L0 = 1e-6

        diag_px = math.sqrt(screen_width_px ** 2 + screen_height_px ** 2)
        ppi = diag_px / screen_diag_inch
        
        pixel_size_m = 0.0254 / ppi # 1 дюйм = 0.0254 метра

        self.phi_d_screen = pixel_size_m / distance_m # считаем угловой размер пикселя на экране
        self.phi_d = self.phi_d_screen
        
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
            phi = self.phi_d

        L0: float = 0.000001
        return (
            L0
            * self._Simkin_A(La)
            * self._Simkin_LAt(self._Simkin_lambda(L, La))
            * self._Simkin_S(self._Simkin_teta(phi, La))
            * self._Simkin_fN(La, L, LN)
            * self._simkin_C(self._simkin_T(t, La))
        )
    
    def find_La_with_background(self, L_patch, optimizer = 'minimize_scalar', target_width_cm = None, draw_plots = False):
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
            self.phi_d = scale * self.phi_d_screen
        else:
            n_patch_screen = 0
            self.phi_d = self.phi_d_screen

        n_total = self.screen_width_px * self.screen_height_px
        
        n_bg = n_total - n_patch_screen

        logL = np.log(np.maximum(L_patch, 1e-12))
        # log_La_min = float(min(logL.min(), np.log(self.l_gray)))
        # log_La_max = float(max(logL.max(), np.log(self.l_gray)))
        log_La_min = -3.0 # La = 0.05
        log_La_max = 6.0 # La = 403.43

        def objective_log_La(log_La):
            curr_La = math.exp(float(log_La))
        
            # Считаем для пикселей патча
            if L_patch.size > 0:
                num_patch = self._simkin_core(L = L_patch, La = curr_La)
                den_patch = self._simkin_core(L = L_patch, La = L_patch)
                sum_patch = (n_patch_screen / L_patch.size) * float(np.sum(np.log(num_patch / den_patch)))
            else:
                sum_patch = 0.0
            
            # 2. Для фона считаем одно число и умножаем на количество пикселей фона
            num_bg = self._simkin_core(self.l_gray, curr_La)
            den_bg = self._simkin_core(self.l_gray, self.l_gray)
            sum_bg = n_bg * math.log(num_bg / den_bg)
            
            return float((sum_patch + sum_bg) / n_total) # считаем фон и саму картинку равнозначными

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

            if draw_plots:
                grid_x = np.linspace(log_La_min, log_La_max, 300)
                grid_loss = [objective_log_La(x) for x in grid_x]

                plt.figure(figsize=(10, 5))
                plt.plot(
                    grid_x,
                    grid_loss,
                    label='Loss Landscape',
                    color='gray',
                    linewidth=2,
                )
                plt.plot(
                    history_x,
                    history_loss,
                    color='steelblue',
                    linestyle='--',
                    alpha=0.6,
                    label='Search Path',
                )

                scatter = plt.scatter(
                    history_x,
                    history_loss,
                    c=range(len(history_x)),
                    cmap='viridis',
                    s=45,
                    zorder=3,
                    label='Evaluated Points',
                )
                plt.scatter(
                    result.x,
                    result.fun,
                    color='red',
                    marker='*',
                    s=160,
                    zorder=4,
                    label=f'Optimal La = {self.La:.2f}',
                )

                plt.colorbar(scatter, label='Step Index')
                plt.xlabel('log(La)')
                plt.ylabel('Loss')
                plt.title('Optimization Landscape and Trajectory')
                plt.grid(True, linestyle=':', alpha=0.6)
                plt.legend()
                plt.show()

        elif optimizer == 'log_grid_search':

            log_space = np.linspace(-3, 6, num=1000)

            best_la = 0
            min_loss = 1e10

            for log_la in log_space:
                loss = objective_log_La(log_la)
                if min_loss > loss:
                    min_loss = loss
                    best_la = np.exp(log_la)

            self.La = best_la
        
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
        return np.searchsorted(self.bounds, L, side = 'right') - 1 - self.len_left
    
    def k_to_L(self, k): # всегда берем нижнюю границу интервала, в который попали
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

    def sRGB_Companding(self, linear_image):
        linear = np.clip(np.asarray(linear_image, dtype=np.float64), 0.0, 1.0)
        
        encoded = np.where(
            linear <= 0.0031308,
            linear * 12.92,
            1.055 * (linear ** (1.0 / 2.4)) - 0.055
        )
        return encoded

    def Inverse_sRGB_Companding(self, image):
        img = np.asarray(image, dtype=np.float64)
        
        linear = np.where(
            img <= 0.04045,
            img / 12.92,
            ((img + 0.055) / 1.055) ** 2.4
        )
        return linear

    def Linear_RGB_to_XYZ(self, image):
        XYZ_image = image.reshape(-1, 3)
        return (XYZ_image @ self.M.T).reshape(image.shape)

    def XYZ_to_xyY(self, XYZ_image, L_MAX=300.0, coord_white=(0.3127, 0.3290)):
        sum_xyz = XYZ_image.sum(axis=-1, keepdims=True)
        eps = 1e-10
        xy = XYZ_image[..., :2] / (sum_xyz + eps)
        is_black = (sum_xyz.squeeze(-1) < eps)
        if np.any(is_black):
            xy[is_black] = coord_white
        Y = XYZ_image[..., 1]
        L_norm = np.clip(Y, 0.0, 1.0)
        return np.stack([xy[..., 0], xy[..., 1], L_norm], axis=-1)

    def XYZ_to_xzL(self, XYZ_image, L_MAX=300.0, coord_white=(0.3127, 0.3583)):
        sum_xyz = XYZ_image.sum(axis=-1)
        eps = 1e-10
        x = XYZ_image[..., 0] / (sum_xyz + eps)
        z = XYZ_image[..., 2] / (sum_xyz + eps)
        is_black = (sum_xyz < eps)
        if np.any(is_black):
            x[is_black] = coord_white[0]
            z[is_black] = coord_white[1]
        Y = XYZ_image[..., 1]
        L_norm = np.clip(Y, 0.0, 1.0)
        return np.stack([x, z, L_norm], axis=-1)

    def RGB_to_xyL(self, img, L_MAX=300.0,):
        
        V = self.read_img(img)
    
        # [0, 1]
        norm_img = self.normalize_image(V)
    
        # sRGB -> linRGB
        linRGBimg = self.Inverse_sRGB_Companding(norm_img)
    
        #linRGB -> XYZ
        XYZ_image = self.Linear_RGB_to_XYZ(linRGBimg)
    
        xyL_image = self.XYZ_to_xyY(XYZ_image, L_MAX)

        return xyL_image

    def L_to_RGB(self, L, L_MIN = 0.1, L_MAX = 300):
        '''
        L - число или матрица ненормированных яркостей
        '''

        Y_lin = (L - L_MIN) / (L_MAX - L_MIN)

        Y_lin = np.clip(Y_lin, 0.0, 1.0)

        V = self.sRGB_Companding(Y_lin)

        image = np.rint(V * 255).astype(np.uint8)

        return image

class Cifar10Dataset(Dataset):
    CIFAR10_CLASSES = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]

    def __init__(self, data_dir, is_jnd=False, layers=None, is_train=True):
        self.data_dir = Path(data_dir)
        self.is_jnd = is_jnd
        self.layers = layers
        self.is_train = is_train

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Папка {self.data_dir} не найдена")

        self.classes = self.CIFAR10_CLASSES
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        self.samples = []
        extension = "*.npy" if is_jnd else "*.png"

        for cls_name in self.classes:
            cls_folder = self.data_dir / cls_name
            if not cls_folder.exists():
                print(f"Предупреждение: папка {cls_folder} отсутствует, пропускаем")
                continue

            class_idx = self.class_to_idx[cls_name]

            if is_jnd:
                file_paths = glob.glob(str(cls_folder / extension))
            else:
                file_paths = glob.glob(str(cls_folder / "*.png"))
                if not file_paths:
                    file_paths = glob.glob(str(cls_folder / "*.jpg"))

            for path in file_paths:
                self.samples.append((path, class_idx))

        if not self.samples:
            raise RuntimeError(f"Не найдено ни одного файла в {data_dir}")

    def __len__(self):
        return len(self.samples)
    
    
    def apply_augmentations(self, image):
        if not self.is_train:
            return image

        if torch.rand(1) > 0.5:
            image = TF.hflip(image)
            
        angle = random.uniform(-10, 10)
        
        image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.NEAREST)
        
        image = TF.pad(image, padding=4, padding_mode='reflect')
        
        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(32, 32))
        
        image = TF.crop(image, i, j, h, w)
        
        return image

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]

        if self.is_jnd:
            jnd_matrix = np.load(file_path).astype(np.float32)
            image = torch.from_numpy(jnd_matrix).permute(2, 0, 1)
        else:
            image = Image.open(file_path).convert('RGB')
            image = TF.to_tensor(image)

        image = self.apply_augmentations(image)

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

def train_eval(model, optimizer, criterion, scheduler, epochs, train_loader, val_loader, model_name, device):
    loses_history = []
    metric_history = []
    best_acc = 0.0
    scaler = torch.amp.GradScaler('cuda')
    weights_path = '/home/misavinov/weights'

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0

        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]'):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / len(train_loader)
        loses_history.append(avg_train_loss)

        model.eval()
        correct_predictions = 0
        total_samples = 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]'):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, dim=1)
                correct_predictions += torch.sum(preds == labels).item()
                total_samples += labels.size(0)

        epoch_accuracy = correct_predictions / total_samples
        metric_history.append(epoch_accuracy)
        scheduler.step()

        if epoch_accuracy > best_acc:
            best_acc = epoch_accuracy
            torch.save(model.state_dict(), os.path.join(weights_path, f'{model_name}_weight.pth'))

        print(f'Epoch: {epoch + 1}/{epochs}. Train loss: {avg_train_loss:.4f}. Accuracy: {epoch_accuracy:.4f}')

    model.load_state_dict(torch.load(os.path.join(weights_path, f'{model_name}_weight.pth')))
    return loses_history, metric_history