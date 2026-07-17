import os
import math
import glob
import random
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from scipy.stats import norm
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
import torchattacks
from tqdm.auto import tqdm

class JNDModel:
    def __init__(self, L_min, L_max, t=1, phi_d=1e-3, p=0.75):
        self.L_min = L_min
        self.L_max = L_max
        self.t = t
        self.phi_d = phi_d
        self.p = p
        self.L0 = 1e-6

    def _K(self, lambd, b1=0.98, b2=0.1, lambd1=2e-2):
        lambd = np.asarray(lambd, dtype=np.float64)
        return np.where(lambd <= 1, b1 * (1 + lambd1 / lambd), lambd ** b2)

    def _A(self, La, a1=4.8e6, a2=7.1e3, a3=4.5e-4):
        return 1 + (a1 * La) ** 0.5 + a2 * La * (1 + (a3 * La) ** 0.5)

    def _Lambda(self, lambd):
        return lambd * self._K(lambd)

    def _eta(self, La, L3=1e6, L4=1, a4=600):
        return 1 + L3 / (1 + a4 * La * (L4 + La))

    def _C(self, La, t):
        tao_min = 0.05
        T = t / (tao_min * self._eta(La))
        return 1 + 1 / T

    def _func_phi_1(self, La, c2=0.25, L1=10, L2=2.6e-5):
        return (1 + L1 / (L2 + La)) ** c2

    def _S(self, La, theta_1=6.1, a_d=2, inf_phi_1=1):
        phi_0_min = 0.5 * (1 / 60) * (np.pi / 180)
        phi_1 = self._func_phi_1(La)
        phi_0 = phi_1 * phi_0_min / inf_phi_1
        return (theta_1 * phi_0 / self.phi_d + 1.0) ** a_d

    def _l(self, La):
        return self.L0 * self._A(La)

    def _P(self, p, p_ref=0.75):
        return abs(norm.ppf(p) / norm.ppf(p_ref))

    def _D(self, L, La):
        lambd = L / (10 ** 2 * self._l(La))
        return self.L0 * self._A(La) * self._C(La, self.t) * self._Lambda(lambd) * self._S(La, self.phi_d) * self._P(self.p)

    def find_La(self, L_matrix):
        start_La = np.mean(L_matrix)
        EPS = 1e-12
        logL = np.log(np.maximum(L_matrix, EPS))
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
        self.La = math.exp(float(result.x))
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
        return np.searchsorted(self.bounds, L, side='right') - 1 - self.len_left

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
        jnd_model.find_La(L_physical)
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