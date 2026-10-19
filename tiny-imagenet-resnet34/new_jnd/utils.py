import os
import math
import glob
import random
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from scipy.stats import norm
from scipy.optimize import minimize_scalar, minimize
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
import torchattacks
from tqdm.auto import tqdm


class HistogramSimkinJNDModel:
    def __init__(
        self,
        l_min: float = 0.1,
        l_max: float = 300.0,
        screen_diag_inch: float = 16.0,
        screen_width_px: int = 1920,
        screen_height_px: int = 1080,
        distance_m: float = 0.5,
        l_gray: float = 60.0,
        exposure_time_s: float = 1.0,
        noise_luminance: float = 0.0,
    ) -> None:
        self.l_min = float(l_min)
        self.l_max = float(l_max)
        self.l0 = 1e-6
        self.l_gray = float(l_gray)
        self.exposure_time_s = float(exposure_time_s)
        self.noise_luminance = float(noise_luminance)

        self.screen_width_px = int(screen_width_px)
        self.screen_height_px = int(screen_height_px)

        diag_px = math.hypot(self.screen_width_px, self.screen_height_px)
        self.ppi = diag_px / screen_diag_inch

        pixel_size_m = 0.0254 / self.ppi
        self.phi_screen = pixel_size_m / distance_m
        self.phi = self.phi_screen

        self.la = self.l_gray
        self.bounds = np.empty(0, dtype=np.float64)
        self.len_left = 0

    @staticmethod
    def _simkin_a(la: float | np.ndarray) -> float | np.ndarray:
        a1 = 4800000.0
        a2 = 7100.0
        a3 = 0.00045
        return 1.0 + np.sqrt(a1 * la) + a2 * la * (1.0 + np.sqrt(a3 * la))

    def _simkin_l(self, la: float | np.ndarray) -> float | np.ndarray:
        return self.l0 * self._simkin_a(la)

    def _simkin_lambda(
        self, l_val: float | np.ndarray, la: float | np.ndarray
    ) -> float | np.ndarray:
        xao = 100.0
        return l_val / (xao * self._simkin_l(la))

    @staticmethod
    def _simkin_lat(lambda_: float | np.ndarray) -> float | np.ndarray:
        b1 = 0.98
        b2 = 0.1
        lambda1 = 0.02
        lambda_arr = np.asarray(lambda_, dtype=np.float64)
        res = np.where(lambda_arr <= 1.0, b1 * (lambda_arr + lambda1), lambda_arr ** (1.0 + b2))
        if np.ndim(lambda_) == 0:
            return float(res)
        return res

    @staticmethod
    def _simkin_ro(la: float | np.ndarray) -> float | np.ndarray:
        l1 = 10.0
        l2 = 0.000026
        c2 = 0.25
        return (1.0 + l1 / (l2 + la)) ** c2

    def _simkin_teta(self, phi: float, la: float | np.ndarray) -> float | np.ndarray:
        phi_omin = (1.0 / 60.0) * np.pi / 180.0
        return phi / (phi_omin * self._simkin_ro(la))

    @staticmethod
    def _simkin_s(teta: float | np.ndarray) -> float | np.ndarray:
        teta1 = 6.1
        return (teta1 / teta + 1.0) ** 2

    @staticmethod
    def _simkin_nu(la: float | np.ndarray) -> float | np.ndarray:
        nu = 4.1 - 3.4 / (1.0 + np.exp(-0.5 - np.log10(np.maximum(la, 1e-12))))
        return np.clip(nu, 1.0, 4.0)

    def _simkin_t(self, t: float, la: float | np.ndarray) -> float | np.ndarray:
        return t / (0.05 * self._simkin_nu(la))

    @staticmethod
    def _simkin_c(t_val: float | np.ndarray) -> float | np.ndarray:
        return 1.0 + 1.0 / t_val

    def _simkin_fn(
        self, la: float | np.ndarray, l_val: float | np.ndarray, ln: float
    ) -> float | np.ndarray:
        sigma_n = 3.0 * ln / self._simkin_l(la)
        lat_val = self._simkin_lat(self._simkin_lambda(l_val, la))
        return np.sqrt(1.0 + (sigma_n * sigma_n) / (lat_val * lat_val))

    def compute_jnd(
        self,
        l_val: float | np.ndarray,
        la: float | np.ndarray | None = None,
        phi: float | None = None,
        ln: float | None = None,
        t: float | None = None,
    ) -> float | np.ndarray:
        curr_la = self.la if la is None else la
        curr_phi = self.phi if phi is None else phi
        curr_ln = self.noise_luminance if ln is None else ln
        curr_t = self.exposure_time_s if t is None else t

        safe_la = np.maximum(curr_la, 1e-12)
        safe_l = np.maximum(l_val, 1e-12)

        return (
            self.l0
            * self._simkin_a(safe_la)
            * self._simkin_lat(self._simkin_lambda(safe_l, safe_la))
            * self._simkin_s(self._simkin_teta(curr_phi, safe_la))
            * self._simkin_fn(safe_la, safe_l, curr_ln)
            * self._simkin_c(self._simkin_t(curr_t, safe_la))
        )

    def find_la(
        self,
        patch: np.ndarray,
        target_width_cm: float | None = None,
        method: str = "scalar",
        loss_type: str = "l2",
    ) -> float:
        patch_arr = np.asarray(patch, dtype=np.float64)
        if patch_arr.ndim == 2 and patch_arr.size > 0:
            h_patch_px, w_patch_px = patch_arr.shape
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

        if patch_arr.size > 0:
            log_l = np.log(np.maximum(patch_arr, 1e-12))
            log_la_min = float(min(log_l.min(), math.log(self.l_gray)))
            log_la_max = float(max(log_l.max(), math.log(self.l_gray)))
            den_patch = self.compute_jnd(l_val=patch_arr, la=patch_arr)
        else:
            log_la_min = -3.0
            log_la_max = 6.0
            den_patch = np.empty(0, dtype=np.float64)

        if log_la_min == log_la_max:
            log_la_min -= 1.0
            log_la_max += 1.0

        den_bg = self.compute_jnd(l_val=self.l_gray, la=self.l_gray)

        def objective(log_la_val: float) -> float:
            curr_la = math.exp(float(log_la_val))
            if patch_arr.size > 0:
                num_patch = self.compute_jnd(l_val=patch_arr, la=curr_la)
                ratio_patch = np.maximum(num_patch / den_patch, 1e-12)
                log_ratio_patch = np.log(ratio_patch)
                if loss_type == "l1":
                    sum_patch = (n_patch_screen / patch_arr.size) * float(np.sum(np.abs(log_ratio_patch)))
                else:
                    sum_patch = (n_patch_screen / patch_arr.size) * float(np.sum(log_ratio_patch ** 2))
            else:
                sum_patch = 0.0

            num_bg = self.compute_jnd(l_val=self.l_gray, la=curr_la)
            ratio_bg = max(num_bg / den_bg, 1e-12)
            log_ratio_bg = math.log(ratio_bg)
            if loss_type == "l1":
                sum_bg = n_bg * abs(log_ratio_bg)
            else:
                sum_bg = n_bg * (log_ratio_bg ** 2)

            return float((sum_patch + sum_bg) / n_total)

        if method == "scalar":
            res = minimize_scalar(
                objective,
                bounds=(log_la_min, log_la_max),
                method="bounded",
                options={"xatol": 1e-6},
            )
            self.la = math.exp(float(res.x))
        elif method == "bfgs":
            if patch_arr.size > 0:
                mean_l = (float(np.mean(patch_arr)) * n_patch_screen + self.l_gray * n_bg) / n_total
            else:
                mean_l = self.l_gray
            x0 = math.log(max(mean_l, 1e-12))
            res = minimize(
                lambda x: objective(x[0]),
                x0=np.array([x0]),
                bounds=[(log_la_min, log_la_max)],
                method="L-BFGS-B",
                options={"ftol": 1e-9},
            )
            self.la = math.exp(float(res.x[0]))
        elif method == "grid":
            grid = np.linspace(log_la_min, log_la_max, 1000)
            best_loss = float("inf")
            best_la = self.l_gray
            for val in grid:
                current_loss = objective(val)
                if current_loss < best_loss:
                    best_loss = current_loss
                    best_la = math.exp(val)
            self.la = best_la
        else:
            raise ValueError(f"Unknown optimization method: {method}")

        return self.la

    def build_level_boundaries(self) -> np.ndarray:
        right_bounds: list[float] = []
        curr_l = self.la
        while curr_l < self.l_max:
            step = float(self.compute_jnd(l_val=curr_l, la=self.la))
            if step <= 0:
                break
            curr_l += step
            right_bounds.append(min(curr_l, self.l_max))

        left_bounds: list[float] = []
        curr_l = self.la
        while curr_l > self.l0:
            step = float(self.compute_jnd(l_val=curr_l, la=self.la))
            if step <= 0:
                break
            curr_l -= step
            left_bounds.append(max(curr_l, self.l0))

        self.len_left = len(left_bounds)
        left_bounds.reverse()
        self.bounds = np.array(left_bounds + [self.la] + right_bounds, dtype=np.float64)
        return self.bounds

    def l_to_k(self, l_val: float | np.ndarray) -> int | np.ndarray:
        idx = np.searchsorted(self.bounds, l_val, side="right") - 1
        idx = np.clip(idx, 0, len(self.bounds) - 1)
        return idx - self.len_left

    def k_to_l(self, k: int | np.ndarray) -> float | np.ndarray:
        indices = np.asarray(k) + self.len_left
        return self.bounds[indices]


JNDModel = HistogramSimkinJNDModel


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
        sum_xyz = XYZ_image.sum(keepdims=True, axis=-1)
        xyz = XYZ_image / (sum_xyz + self.eps)
        is_black = (sum_xyz < self.eps).squeeze(-1) 

        if np.any(is_black):
            xyz[is_black] = self.coord_white

        Y = XYZ_image[..., 1]
        L_norm = np.clip(Y, 0.0, 1.0)
        L_physical = self.L_MIN + L_norm * (self.L_MAX - self.L_MIN)
        L_physical = np.expand_dims(L_physical, axis=-1)
        xyzL = np.concatenate([xyz, L_physical], axis=-1)
        return xyzL

    def RGB_to_xyzL(self, image: np.ndarray) -> np.ndarray:
        norm_img = np.asarray(image, dtype=np.float64) / 255.0
        linRGBimg = self.Inverse_sRGB_Companding(norm_img)
        XYZ_image = self.Linear_RGB_to_XYZ(linRGBimg)
        xyzL_image = self.XYZ_to_xyzL(XYZ_image)
        return xyzL_image

    def L_to_gray(self, L: np.ndarray | float) -> np.ndarray | np.uint8:
        Y_lin = (L - self.L_MIN) / (self.L_MAX - self.L_MIN)
        Y_lin = np.clip(Y_lin, 0.0, 1.0)
        V = self.sRGB_Companding(Y_lin)
        gray_matrix = np.rint(V * 255).astype(np.uint8)
        return gray_matrix


class TinyImageNetDataset(Dataset):
    def __init__(self, data_dir, is_jnd, is_train, layers=None):
        self.layers = layers
        self.is_train = is_train
        self.data_dir = data_dir
        self.is_jnd = is_jnd
        self.samples = []

        self.classes = sorted([d for d in os.listdir(data_dir)
                if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith('.') and d != 'images'])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        for cls_name in self.classes:
            cls_folder = os.path.join(data_dir, cls_name)
            class_idx = self.class_to_idx[cls_name]
            if is_jnd:
                file_paths = glob.glob(os.path.join(cls_folder, "*.npy"))
            else:
                file_paths = glob.glob(os.path.join(cls_folder, "images", "*.JPEG"))
            for path in file_paths:
                self.samples.append((path, class_idx))

    def __len__(self):
        return len(self.samples)

    def apply_augmentations(self, image):
        if not self.is_train:
            return image

        if torch.rand(1) > 0.5:
            image = TF.hflip(image)

        angle = random.uniform(-15, 15)
        translate_x = random.randint(-4, 4)
        translate_y = random.randint(-4, 4)
        scale = random.uniform(0.85, 1.15)
        shear = random.uniform(-5, 5)
        
        image = TF.affine(
            image, angle=angle, translate=[translate_x, translate_y], 
            scale=scale, shear=[shear], interpolation=TF.InterpolationMode.NEAREST
        )

        image = TF.pad(image, padding=8, padding_mode='reflect')
        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(64, 64))
        image = TF.crop(image, i, j, h, w)
        
        if torch.rand(1) > 0.5:
            i, j, h, w, v = transforms.RandomErasing.get_params(
                image, scale=(0.02, 0.15), ratio=(0.3, 3.3)
            )
            image = TF.erase(image, i, j, h, w, v=0)

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


class ResNet34(nn.Module):
    def __init__(self, num_classes=200):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.layer1 = self._make_layer(64, num_blocks=3, stride=1)
        self.layer2 = self._make_layer(128, num_blocks=4, stride=2)
        self.layer3 = self._make_layer(256, num_blocks=6, stride=2)
        self.layer4 = self._make_layer(512, num_blocks=3, stride=2)
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
    weights_path = '/home/misavinov/scratch/ws/my_space/new_jnd/weights'

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


def rgb_tensor_to_tensor(rgb_tensor, converter, jnd_model, L_MIN, L_MAX, layers=None):
    rgb_np = rgb_tensor.permute(0, 2, 3, 1).cpu().numpy().astype(np.float64)
    jnd_batch = []
    
    for i in range(rgb_np.shape[0]):
        img = rgb_np[i]
        norm_img = converter.normalize_image(img)
        linRGBimg = converter.Inverse_sRGB_Companding(norm_img)
        XYZ_image = converter.Linear_RGB_to_XYZ(linRGBimg)

        xyL_image = converter.XYZ_to_xyY(XYZ_image, L_MAX)
        xzL_image = converter.XYZ_to_xzL(XYZ_image, L_MAX)

        x = xyL_image[..., 0]
        z = xzL_image[..., 1]
        L_norm = xyL_image[..., 2]

        L_physical = L_MIN + (L_MAX - L_MIN) * L_norm
        jnd_model.find_la(L_physical)
        jnd_model.build_level_boundaries()
        k_map = jnd_model.L_to_k(L_physical)

        if layers == 'xzk':
            output_matrix = np.stack([x, z, k_map], axis=-1)
        elif layers == 'kkk':
            output_matrix = np.stack([k_map, k_map, k_map], axis=-1)
        else:
            raise ValueError(f"Неизвестный режим слоев: {layers}")

        jnd_batch.append(output_matrix)

    jnd_batch_np = np.stack(jnd_batch, axis=0)
    return torch.from_numpy(jnd_batch_np.astype(np.float32)).permute(0, 3, 1, 2).to(rgb_tensor.device)


def evaluate_single_model_robustness(model, attack_model, attack_class, loader, converter, jnd_model, L_MIN, L_MAX, layers="rgb", **kwargs):
    model.eval()
    attack_model.eval()

    total_clean_correct, total_adv_correct = 0, 0
    total_clean_conf, total_adv_conf = 0.0, 0.0
    total_images = 0

    attack = attack_class(attack_model, **kwargs)
    device = next(model.parameters()).device

    for images, labels in tqdm(loader, desc=f"Атака {attack_class.__name__}", leave=False):
        images, labels = images.to(device), labels.to(device)
        adv_images_rgb = attack(images, labels)

        def prepare_inputs(input_rgb):
            if layers == "rgb":
                return input_rgb
            return rgb_tensor_to_tensor(input_rgb, converter, jnd_model, L_MIN, L_MAX, layers=layers)

        clean_inputs = prepare_inputs(images)
        adv_inputs = prepare_inputs(adv_images_rgb)

        outputs_clean = model(clean_inputs)
        _, predicted_clean = torch.max(outputs_clean.data, 1)
        total_clean_correct += (predicted_clean == labels).sum().item()
        probs_clean = F.softmax(outputs_clean, dim=1)
        total_clean_conf += probs_clean.gather(1, labels.view(-1, 1)).squeeze().sum().item()

        outputs_adv = model(adv_inputs)
        _, predicted_adv = torch.max(outputs_adv.data, 1)
        total_adv_correct += (predicted_adv == labels).sum().item()
        probs_adv = F.softmax(outputs_adv, dim=1)
        total_adv_conf += probs_adv.gather(1, labels.view(-1, 1)).squeeze().sum().item()

        total_images += len(labels)

    return {
        "Clean Acc": 100 * total_clean_correct / total_images,
        "Adv Acc": 100 * total_adv_correct / total_images,
        "Drop": 100 * (total_clean_correct - total_adv_correct) / total_images,
        "Clean Conf": total_clean_conf / total_images,
        "Adv Conf": total_adv_conf / total_images,
    }


def get_results_for_model(model, attack_model, loader, converter, jnd_model, L_MIN, L_MAX, layers="rgb"):
    data = []
    attacks = [torchattacks.FGSM, torchattacks.BIM, torchattacks.PGD, torchattacks.CW]
    names = ["FGSM", "BIM", "PGD", "CW"]
    params = [
        {"eps": 8/255},
        {"eps": 8/255, "alpha": 2/255, "steps": 10},
        {"eps": 8/255, "alpha": 1/255, "steps": 10, "random_start": True},
        {"c": 1, "kappa": 0, "steps": 50, "lr": 0.01}
    ]

    for i, attack_class in tqdm(enumerate(attacks), desc=f'Тестирование режима {layers}', total=4):
        res = evaluate_single_model_robustness(
            model=model, attack_model=attack_model, attack_class=attack_class,
            loader=loader, converter=converter, jnd_model=jnd_model,
            L_MIN=L_MIN, L_MAX=L_MAX, layers=layers, **params[i]
        )
        res["Attack"] = names[i]
        data.append(res)

    return pd.DataFrame(data).set_index("Attack")


def evaluate_epsilon_robustness(model, attack_model, attack_class, loader, converter, jnd_model, L_MIN, L_MAX, layers="rgb", epsilons=None, **base_params):
    model.eval()
    attack_model.eval()
    device = next(model.parameters()).device
    acc_drop_series = []

    for eps in tqdm(epsilons, desc=f"Тестирование eps для {layers}", leave=False):
        if eps == 0.0:
            total_correct = 0
            total_images = 0
            for images, labels in tqdm(loader, desc="Чистый accuracy", leave=False):
                images, labels = images.to(device), labels.to(device)
                clean_inputs = images if layers == "rgb" else rgb_tensor_to_tensor(images, converter, jnd_model, L_MIN, L_MAX, layers=layers)
                with torch.no_grad():
                    outputs = model(clean_inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total_correct += (predicted == labels).sum().item()
                    total_images += len(labels)
            acc_drop_series.append(100 * total_correct / total_images)
            continue

        current_params = base_params.copy()
        if attack_class in [torchattacks.FGSM, torchattacks.BIM, torchattacks.PGD]:
            current_params["eps"] = eps
            if "alpha" in current_params:
                current_params["alpha"] = eps / 4
        elif attack_class == torchattacks.CW:
            if eps > 0:
                current_params["c"] = base_params.get("c", 1) * (eps * 255 / 8)

        attack = attack_class(attack_model, **current_params)
        total_adv_correct = 0
        total_images = 0

        for images, labels in tqdm(loader, desc=f"Атака eps = {eps:.4f}", leave=False):
            images, labels = images.to(device), labels.to(device)
            try:
                adv_images_rgb = attack(images, labels)
            except:
                adv_images_rgb = images

            adv_inputs = adv_images_rgb if layers == "rgb" else rgb_tensor_to_tensor(adv_images_rgb, converter, jnd_model, L_MIN, L_MAX, layers=layers)
            
            with torch.no_grad():
                outputs_adv = model(adv_inputs)
                _, predicted_adv = torch.max(outputs_adv.data, 1)
                total_adv_correct += (predicted_adv == labels).sum().item()
                total_images += len(labels)

        acc_drop_series.append(100 * total_adv_correct / total_images)

    return acc_drop_series


def get_physical_luminance(img_rgb, converter, L_MIN, L_MAX):
    img_input = np.clip(img_rgb, 0, 1).astype(np.float64)
    linRGBimg = converter.Inverse_sRGB_Companding(img_input)
    XYZ_image = converter.Linear_RGB_to_XYZ(linRGBimg)
    Y_relative = XYZ_image[..., 1]
    return L_MIN + (L_MAX - L_MIN) * Y_relative