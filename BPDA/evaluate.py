import os
import random
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchattacks
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Subset
from multiprocessing.pool import ThreadPool

from utils import (
    Cifar10Dataset,
    ImageConverter,
    JNDModel,
    ResNet18
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

BASE_DIR = Path('/scratch/misavinov/BPDA')
DATA_DIR = BASE_DIR / 'data'
JND_DIR = DATA_DIR / 'cifar10_val_jnd'

NUM_WORKERS = int(os.environ.get('SLURM_CPUS_PER_TASK', 4))
L_MIN, L_MAX = 0.1, 300.0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

WEIGHTS_DIR = BASE_DIR / 'weights'
RESULTS_DIR = BASE_DIR / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class JNDBPDAFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_rgb, converter, jnd_model, layer_type):
        x_np = x_rgb.detach().cpu().numpy().transpose(0, 2, 3, 1)

        def process_img(img):
            norm_img = converter.normalize_image(img)
            linRGB = converter.Inverse_sRGB_Companding(norm_img)
            XYZ = converter.Linear_RGB_to_XYZ(linRGB)
            
            xyL = converter.XYZ_to_xyY(XYZ, L_MAX)
            xzL = converter.XYZ_to_xzL(XYZ, L_MAX)

            x = xyL[..., 0]
            y = xyL[..., 1]
            z = xzL[..., 1]

            L_phys = L_MIN + (L_MAX - L_MIN) * xyL[..., 2]
            
            local_jnd = JNDModel(L_MIN, L_MAX)
            local_jnd.find_La(L_phys)
            local_jnd.build_level_boundaries()
            k_map = local_jnd.L_to_k(L_phys)

            if layer_type == 'kkk':
                jnd_img = np.stack([k_map, k_map, k_map], axis=-1)
            elif layer_type == 'xyk':
                jnd_img = np.stack([x, y, k_map], axis=-1)
            elif layer_type == 'xzk':
                jnd_img = np.stack([x, z, k_map], axis=-1)
            else:
                jnd_img = np.stack([x, z, k_map], axis=-1)

            return jnd_img

        with ThreadPool(processes=NUM_WORKERS) as pool:
            batch_jnd = pool.map(process_img, x_np)

        return torch.from_numpy(np.array(batch_jnd)).permute(0, 3, 1, 2).float().to(x_rgb.device)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None


class BPDAModelWrapper(nn.Module):
    def __init__(self, classifier_model, converter, jnd_model, layer_type='xzk'):
        super().__init__()
        self.model = classifier_model
        self.converter = converter
        self.jnd_model = jnd_model
        self.layer_type = layer_type

    def forward(self, x_rgb):
        if self.layer_type == 'rgb':
            return self.model(x_rgb)

        x_jnd = JNDBPDAFunction.apply(x_rgb, self.converter, self.jnd_model, self.layer_type)
        return self.model(x_jnd)


def convert_batch_rgb_to_jnd(x_rgb, converter, jnd_model, layer_type):
    x_np = x_rgb.detach().cpu().numpy().transpose(0, 2, 3, 1)
    batch_jnd = []

    for img in x_np:
        norm_img = converter.normalize_image(img)
        linRGB = converter.Inverse_sRGB_Companding(norm_img)
        XYZ = converter.Linear_RGB_to_XYZ(linRGB)
        xyL = converter.XYZ_to_xyY(XYZ, L_MAX)
        xzL = converter.XYZ_to_xzL(XYZ, L_MAX)

        x = xyL[..., 0]
        y = xyL[..., 1]
        z = xzL[..., 1]

        L_phys = L_MIN + (L_MAX - L_MIN) * xyL[..., 2]
        
        local_jnd = JNDModel(L_MIN, L_MAX)
        local_jnd.find_La(L_phys)
        local_jnd.build_level_boundaries()
        k_map = local_jnd.L_to_k(L_phys)

        if layer_type == 'kkk':
            jnd_img = np.stack([k_map, k_map, k_map], axis=-1)
        elif layer_type == 'xyk':
            jnd_img = np.stack([x, y, k_map], axis=-1)
        elif layer_type == 'xzk':
            jnd_img = np.stack([x, z, k_map], axis=-1)
        else:
            jnd_img = np.stack([x, z, k_map], axis=-1)

        batch_jnd.append(jnd_img)

    return torch.from_numpy(np.array(batch_jnd)).permute(0, 3, 1, 2).float().to(x_rgb.device)


def evaluate_transfer_attack(target_model, attack_instance, rgb_loader, converter, jnd_model, target_layer, device):
    target_model.eval()

    clean_correct, adv_correct, total = 0, 0, 0
    clean_conf_sum, adv_conf_sum = 0.0, 0.0

    for images, labels in tqdm(rgb_loader, desc=f"Transfer -> {target_layer}", leave=False):
        images, labels = images.to(device), labels.to(device)
        adv_rgb_images = attack_instance(images, labels)

        if target_layer == 'rgb':
            clean_input = images
            adv_input = adv_rgb_images
        else:
            clean_input = convert_batch_rgb_to_jnd(images, converter, jnd_model, target_layer)
            adv_input = convert_batch_rgb_to_jnd(adv_rgb_images, converter, jnd_model, target_layer)

        with torch.no_grad():
            clean_outputs = target_model(clean_input)
            clean_probs = F.softmax(clean_outputs, dim=1)
            clean_preds = torch.argmax(clean_outputs, dim=1)
            clean_correct += (clean_preds == labels).sum().item()
            clean_conf_sum += clean_probs.gather(1, labels.unsqueeze(1)).sum().item()

            adv_outputs = target_model(adv_input)
            adv_probs = F.softmax(adv_outputs, dim=1)
            adv_preds = torch.argmax(adv_outputs, dim=1)
            adv_correct += (adv_preds == labels).sum().item()
            adv_conf_sum += adv_probs.gather(1, labels.unsqueeze(1)).sum().item()

        total += labels.size(0)

    clean_acc = 100.0 * clean_correct / total
    adv_acc = 100.0 * adv_correct / total
    drop = clean_acc - adv_acc
    clean_conf = clean_conf_sum / total
    adv_conf = adv_conf_sum / total

    return clean_acc, adv_acc, drop, clean_conf, adv_conf


def evaluate_bpda_attack(bpda_model, attack_class, attack_params, rgb_loader, device):
    bpda_model.eval()
    
    attack_instance = attack_class(bpda_model, **attack_params)

    clean_correct, adv_correct, total = 0, 0, 0
    clean_conf_sum, adv_conf_sum = 0.0, 0.0

    for images, labels in tqdm(rgb_loader, desc=f"BPDA Attack", leave=False):
        images, labels = images.to(device), labels.to(device)

        with torch.no_grad():
            clean_outputs = bpda_model(images)
            clean_probs = F.softmax(clean_outputs, dim=1)
            clean_preds = torch.argmax(clean_outputs, dim=1)
            clean_correct += (clean_preds == labels).sum().item()
            clean_conf_sum += clean_probs.gather(1, labels.unsqueeze(1)).sum().item()

        adv_rgb_images = attack_instance(images, labels)

        with torch.no_grad():
            adv_outputs = bpda_model(adv_rgb_images)
            adv_probs = F.softmax(adv_outputs, dim=1)
            adv_preds = torch.argmax(adv_outputs, dim=1)
            adv_correct += (adv_preds == labels).sum().item()
            adv_conf_sum += adv_probs.gather(1, labels.unsqueeze(1)).sum().item()

        total += labels.size(0)

    clean_acc = 100.0 * clean_correct / total
    adv_acc = 100.0 * adv_correct / total
    drop = clean_acc - adv_acc
    clean_conf = clean_conf_sum / total
    adv_conf = adv_conf_sum / total

    return clean_acc, adv_acc, drop, clean_conf, adv_conf


if __name__ == '__main__':
    converter = ImageConverter()
    jnd_model = JNDModel(L_MIN, L_MAX)

    full_test_dataset = Cifar10Dataset(data_dir=DATA_DIR / 'test', is_jnd=False)

    SEED = 42
    random.seed(SEED)

    indices = random.sample(range(len(full_test_dataset)), 1000)
    subset_test_dataset = Subset(full_test_dataset, indices)

    rgb_test_loader = DataLoader(
        subset_test_dataset,
        batch_size=512, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
    )

    print(f"Используется случайно выбранное подмножество: {len(subset_test_dataset)} картинок.")
    print("Порядок классов в скрипте:", full_test_dataset.class_to_idx)

    weight_dict = {
        'kkk': 'model_3jnd_weight.pth',
        'rgb': 'model_rgb_weight.pth',
        'xzk': 'model_xkz_weight.pth'
    }

    models_dict = {}
    bpda_models_dict = {}

    for layer in ['rgb', 'kkk', 'xzk']:
        model = ResNet18().to(device)
        weight_file = WEIGHTS_DIR / weight_dict[layer]
        if weight_file.exists():
            model.load_state_dict(torch.load(weight_file, map_location=device))
        models_dict[layer] = model
        bpda_models_dict[layer] = BPDAModelWrapper(model, converter, jnd_model, layer_type=layer).to(device)

    rgb_model = models_dict['rgb']

    attacks_config = {
        'FGSM': (torchattacks.FGSM, {'eps': 8/255}),
        'BIM': (torchattacks.BIM, {'eps': 8/255, 'alpha': 2/255, 'steps': 10}),
        'PGD': (torchattacks.PGD, {'eps': 8/255, 'alpha': 2/255, 'steps': 10, 'random_start': True}),
        'CW': (torchattacks.CW, {'c': 1, 'kappa': 0, 'steps': 50, 'lr': 0.01})
    }

    print("\nЭксперимент 1: Transfer Attack (Шум с RGB)")
    transfer_tables = [RESULTS_DIR / f'table_transfer_{layer}.csv' for layer in ['rgb', 'kkk', 'xzk']]
    if all(table_path.exists() for table_path in transfer_tables):
        print("Пропуск Эксперимента 1: таблицы уже существуют.")
    else:
        for target_layer in ['rgb', 'kkk', 'xzk']:
            csv_path = RESULTS_DIR / f'table_transfer_{target_layer}.csv'
            if csv_path.exists():
                print(f"Таблица transfer для {target_layer} уже существует, пропускаем.")
                continue

            target_model = models_dict[target_layer]
            table_rows = []

            for attack_name, (attack_class, attack_params) in attacks_config.items():
                attack_instance = attack_class(rgb_model, **attack_params)

                clean_acc, adv_acc, drop, clean_conf, adv_conf = evaluate_transfer_attack(
                    target_model=target_model,
                    attack_instance=attack_instance,
                    rgb_loader=rgb_test_loader,
                    converter=converter,
                    jnd_model=jnd_model,
                    target_layer=target_layer,
                    device=device
                )

                table_rows.append({
                    'Attack': attack_name,
                    'Clean Acc': round(clean_acc, 1),
                    'Adv Acc': round(adv_acc, 1),
                    'Drop': round(drop, 1),
                    'Clean Conf': round(clean_conf, 6),
                    'Adv Conf': round(adv_conf, 6)
                })

            df = pd.DataFrame(table_rows).set_index('Attack')
            df.to_csv(csv_path)
            print(f"Сохранена таблица transfer для {target_layer}")

    print("\nЭкcперимент 2: Direct BPDA Attack")
    bpda_tables = [RESULTS_DIR / f'table_bpda_{layer}.csv' for layer in ['rgb', 'kkk', 'xzk']]
    if all(table_path.exists() for table_path in bpda_tables):
        print("Пропуск Эксперимента 2: таблицы уже существуют.")
    else:
        for target_layer in ['rgb', 'kkk', 'xzk']:
            csv_path = RESULTS_DIR / f'table_bpda_{target_layer}.csv'
            if csv_path.exists():
                print(f"Таблица BPDA для {target_layer} уже существует, пропускаем.")
                continue

            bpda_model = bpda_models_dict[target_layer]
            table_rows = []

            for attack_name, (attack_class, attack_params) in attacks_config.items():
                clean_acc, adv_acc, drop, clean_conf, adv_conf = evaluate_bpda_attack(
                    bpda_model=bpda_model,
                    attack_class=attack_class,
                    attack_params=attack_params,
                    rgb_loader=rgb_test_loader,
                    device=device
                )

                table_rows.append({
                    'Attack': attack_name,
                    'Clean Acc': round(clean_acc, 1),
                    'Adv Acc': round(adv_acc, 1),
                    'Drop': round(drop, 1),
                    'Clean Conf': round(clean_conf, 6),
                    'Adv Conf': round(adv_conf, 6)
                })

            df = pd.DataFrame(table_rows).set_index('Attack')
            df.to_csv(csv_path)
            print(f"Сохранена таблица BPDA для {target_layer}")