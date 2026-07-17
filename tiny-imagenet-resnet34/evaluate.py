import os
import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import torchattacks
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Subset
import random
import numpy as np
from utils import TinyImageNetDataset, ResNet34, ImageConverter, JNDModel, rgb_tensor_to_tensor, get_results_for_model, evaluate_epsilon_robustness

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = int(os.environ.get('SLURM_CPUS_PER_TASK', 4))
L_MIN, L_MAX = 0.1, 300.0

weights_path = '/home/misavinov/weights'
results = '/home/misavinov/results'
adv_plots_path = '/home/misavinov/adv_plots'
data_dir = '/scratch/misavinov/data/tiny-imagenet-200'

os.makedirs(results, exist_ok=True)
os.makedirs(adv_plots_path, exist_ok=True)

if __name__ == '__main__':
    converter = ImageConverter()
    jnd_model = JNDModel(L_MIN, L_MAX)

    val_dataset_rgb = TinyImageNetDataset(f'{data_dir}/val', is_jnd=False, is_train=False)
    
    random.seed(42)
    indices = list(random.sample(range(10000), 1000))
    subset_dataset = Subset(val_dataset_rgb, indices)
    val_loader_1000 = DataLoader(subset_dataset, batch_size=128, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    models_dict = {}
    for layer in ['rgb', 'kkk', 'xzk']:
        model = ResNet34().to(device)
        model.load_state_dict(torch.load(os.path.join(weights_path, f'model_{layer}_weight.pth')))
        models_dict[layer] = model

    for layer, model in models_dict.items():
        print(f"Сбор базовых метрик для {layer}...")
        df = get_results_for_model(model, models_dict['rgb'], val_loader_1000, converter, jnd_model, L_MIN, L_MAX, layers=layer)
        df.to_csv(os.path.join(results, f'model_{layer}.csv'), index=True)

    eps_values = [0, 1/255, 2/255, 3/255, 4/255, 5/255, 8/255, 10/255, 16/255, 20/255]
    
    attacks_list = [
        (torchattacks.FGSM, {}),
        (torchattacks.BIM, {"steps": 10, "alpha": 2/255}),
        (torchattacks.PGD, {"steps": 10, "alpha": 1/255, "random_start": True}),
        (torchattacks.CW, {"c": 1, "kappa": 0, "steps": 50, "lr": 0.01})
    ]

    all_results = {}
    
    for attack_class, base_attack_params in attacks_list:
        attack_name = attack_class.__name__
        print(f"Атака: {attack_name}")
        all_results[attack_name] = {}

        for layer_type, current_model in tqdm(models_dict.items(), desc=f"Модели под {attack_name}"):
            accuracies = evaluate_epsilon_robustness(
                model=current_model,
                attack_model=current_model,
                attack_class=attack_class,
                loader=val_loader_1000,
                converter=converter,
                jnd_model=jnd_model,
                L_MIN=L_MIN,
                L_MAX=L_MAX,
                layers=layer_type,
                epsilons=eps_values,
                **base_attack_params,
            )
            all_results[attack_name][layer_type] = accuracies

    for attack_name in all_results.keys():
        plt.figure(figsize=(10, 6), dpi=150)
        linestyles = ["-", "--", "-."]
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
        eps_labels = [f"{int(round(e*255))}/255" if e > 0 else "0" for e in eps_values]

        for i, (layer_type, acc_list) in enumerate(all_results[attack_name].items()):
            plt.plot(
                eps_values,
                acc_list,
                marker=".",
                linestyle=linestyles[i % len(linestyles)],
                color=colors[i % len(colors)],
                linewidth=2,
                markersize=7,
                label=f"Модель: {layer_type}",
            )

        plt.title(f"Устойчивость моделей к атаке {attack_name} при разных eps", fontsize=14, pad=15)
        plt.xlabel("Сила возмущения eps", fontsize=12)
        plt.ylabel("Accuracy (%)", fontsize=12)
        plt.xticks(eps_values, eps_labels, rotation=45)
        plt.ylim(-5, 105)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(fontsize=11, loc="upper right")
        plt.tight_layout()

        plt.savefig(os.path.join(adv_plots_path, attack_name))
        plt.close()