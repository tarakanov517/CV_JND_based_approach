import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import os
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from utils.auxiliary_functions import STL10Dataset, make_backbone, train_eval

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--layer', type=str, required=True, choices=['rgb', 'kkk', 'xzk'], help='Какую модель обучать')
parser.add_argument('--target_width_cm', type=str, required=True, choices=['None', '4', '8', '16'], help='На каком размере рассчитывать La')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = int(os.environ.get('SLURM_CPUS_PER_TASK', 4))
epochs = 100

BASE_DIR = Path('/home/misavinov/scratch/ws/my_space/stl10-resnet50')
weights_path = BASE_DIR / 'weights'
plots_path = BASE_DIR / 'training_plots'

weights_path.mkdir(parents=True, exist_ok=True)
plots_path.mkdir(parents=True, exist_ok=True)

stl = load_dataset("jxie/stl10")

if __name__ == '__main__':

    is_jnd = args.layer != 'rgb'

    if is_jnd:
        jnd_dataset_train = BASE_DIR / 'stl10_xyzk' / args.target_width_cm / 'train_jnd.npy'
        jnd_dataset_val = BASE_DIR / 'stl10_xyzk' / args.target_width_cm / 'test_jnd.npy'
    else:
        jnd_dataset_train = None
        jnd_dataset_val = None

    train_dataset = STL10Dataset(
        hf_dataset=stl['train'],
        matrix_path=jnd_dataset_train,
        is_jnd=is_jnd,
        layers=args.layer,
        is_train=True,
    )
    val_dataset = STL10Dataset(
        hf_dataset=stl['test'],
        matrix_path=jnd_dataset_val,
        is_jnd=is_jnd,
        layers=args.layer,
        is_train=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    model = make_backbone().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    if args.layer == 'rgb':
        model_name = 'model_rgb'
    else:
        suffix = f"{args.target_width_cm}cm" if args.target_width_cm != "None" else "None"
        model_name = f"model_{args.layer}_{suffix}"

    loses_history, metric_history = train_eval(
        model, optimizer, criterion, scheduler, epochs, train_loader, val_loader, model_name, device
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    epochs_range = range(1, len(loses_history) + 1)

    ax1.plot(epochs_range, loses_history, color='blue', label='Train Loss')
    ax1.set_title('Train Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs_range, metric_history, color='red', label='Val acc')
    ax2.set_title('Val Acc')
    ax2.legend()
    ax2.grid(True)

    plt.savefig(plots_path / f"{model_name}.png")
    plt.close()