import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision.transforms.functional as TF
import torchattacks

from utils import (
    ResNet34,
    ImageConverter,
    HistogramSimkinJNDModel,
    TinyImageNetDataset,
)

BASE_DIR = '/home/misavinov/scratch/ws/my_space/new_jnd'
WEIGHTS_DIR = os.path.join(BASE_DIR, 'weights')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
DATA_DIR = os.path.join(BASE_DIR, 'data', 'tiny-imagenet-200', 'val')

os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
L_MIN, L_MAX = 0.1, 300.0


def local_rgb_tensor_to_jnd(rgb_tensor, converter, jnd_model, layers='rgb'):
    if layers == 'rgb':
        return rgb_tensor

    rgb_np = rgb_tensor.detach().permute(0, 2, 3, 1).cpu().numpy().astype(np.float64)
    jnd_batch = []

    for i in range(rgb_np.shape[0]):
        img = rgb_np[i]
        if img.max() <= 1.0:
            img = img * 255.0

        xyzL = converter.RGB_to_xyzL(img)
        x = xyzL[..., 0]
        z = xyzL[..., 2]
        L_physical = xyzL[..., 3]

        jnd_model.find_la(L_physical)
        jnd_model.build_level_boundaries()
        k_map = jnd_model.l_to_k(L_physical)

        if layers == 'kkk':
            out = np.stack([k_map, k_map, k_map], axis=-1)
        elif layers == 'xzk':
            out = np.stack([x, z, k_map], axis=-1)
        else:
            raise ValueError(f"Неизвестный тип слоя: {layers}")

        jnd_batch.append(out)

    jnd_batch_np = np.stack(jnd_batch, axis=0)
    return torch.from_numpy(jnd_batch_np.astype(np.float32)).permute(0, 3, 1, 2).to(DEVICE)


def evaluate_attack(model, attack_model, attack_instance, loader, converter, jnd_model, layers='rgb'):
    model.eval()
    attack_model.eval()

    total_clean_correct = 0
    total_adv_correct = 0
    total_clean_conf = 0.0
    total_adv_conf = 0.0
    total_images = 0

    for images, labels in tqdm(loader, desc=f"Атака {attack_instance.__class__.__name__}", leave=False):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        with torch.no_grad():
            clean_inputs = local_rgb_tensor_to_jnd(images, converter, jnd_model, layers=layers)
            outputs_clean = model(clean_inputs)
            _, preds_clean = torch.max(outputs_clean, 1)
            total_clean_correct += (preds_clean == labels).sum().item()
            probs_clean = F.softmax(outputs_clean, dim=1)
            total_clean_conf += probs_clean.gather(1, labels.view(-1, 1)).squeeze().sum().item()

        adv_images_rgb = attack_instance(images, labels)

        with torch.no_grad():
            adv_inputs = local_rgb_tensor_to_jnd(adv_images_rgb, converter, jnd_model, layers=layers)
            outputs_adv = model(adv_inputs)
            _, preds_adv = torch.max(outputs_adv, 1)
            total_adv_correct += (preds_adv == labels).sum().item()
            probs_adv = F.softmax(outputs_adv, dim=1)
            total_adv_conf += probs_adv.gather(1, labels.view(-1, 1)).squeeze().sum().item()

        total_images += labels.size(0)

    clean_acc = 100.0 * total_clean_correct / total_images
    adv_acc = 100.0 * total_adv_correct / total_images
    drop = clean_acc - adv_acc
    clean_conf = total_clean_conf / total_images
    adv_conf = total_adv_conf / total_images

    return {
        "Clean Acc": clean_acc,
        "Adv Acc": adv_acc,
        "Drop": drop,
        "Clean Conf": clean_conf,
        "Adv Conf": adv_conf,
    }


def main():
    converter = ImageConverter(L_MIN=L_MIN, L_MAX=L_MAX)
    jnd_model = HistogramSimkinJNDModel(l_min=L_MIN, l_max=L_MAX)

    val_dataset = TinyImageNetDataset(data_dir=DATA_DIR, is_jnd=False, is_train=False)

    sample_size = min(500, len(val_dataset))
    indices = list(range(sample_size))
    subset_dataset = Subset(val_dataset, indices)
    val_loader = DataLoader(subset_dataset, batch_size=16, shuffle=False, num_workers=0)

    models_dict = {}
    for layer in ['rgb', 'kkk', 'xzk']:
        model = ResNet34(num_classes=200).to(DEVICE)
        weight_path = os.path.join(WEIGHTS_DIR, f'model_{layer}_weight.pth')

        print(f"Загрузка весов для {layer}: {weight_path}")
        model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
        model.eval()
        models_dict[layer] = model

    attack_model = models_dict['rgb']

    for layer in ['rgb', 'kkk', 'xzk']:
        print(f"\nЗапуск тестирования для режима {layer}...")
        current_model = models_dict[layer]

        attacks = [
            torchattacks.FGSM(attack_model, eps=8/255),
            torchattacks.BIM(attack_model, eps=8/255, alpha=2/255, steps=10),
            torchattacks.PGD(attack_model, eps=8/255, alpha=1/255, steps=10, random_start=True),
            torchattacks.CW(attack_model, c=1, kappa=0, steps=30, lr=0.01)
        ]
        attack_names = ["FGSM", "BIM", "PGD", "CW"]

        results = []
        for name, atk in zip(attack_names, attacks):
            print(f"  Выполняется {name} для {layer}...")
            res = evaluate_attack(
                model=current_model,
                attack_model=attack_model,
                attack_instance=atk,
                loader=val_loader,
                converter=converter,
                jnd_model=jnd_model,
                layers=layer
            )
            res["Attack"] = name
            results.append(res)

        df = pd.DataFrame(results).set_index("Attack")
        out_path = os.path.join(RESULTS_DIR, f'results_{layer}.csv')
        df.to_csv(out_path)
        print(f"Результаты для {layer} сохранены в {out_path}:")
        print(df)


if __name__ == '__main__':
    main()