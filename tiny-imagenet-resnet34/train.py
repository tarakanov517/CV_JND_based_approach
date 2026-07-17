import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from utils import TinyImageNetDataset, ResNet34, train_eval

parser = argparse.ArgumentParser()
parser.add_argument('--layer', type=str, required=True, choices=['rgb', 'kkk', 'xzk'], help='Какую модель обучать')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = int(os.environ.get('SLURM_CPUS_PER_TASK', 4))
epochs = 100

data_dir = '/scratch/misavinov/data/tiny-imagenet-200'
jnd_datasets_train = '/scratch/misavinov/jnd_datasets_train'
jnd_datasets_val = '/scratch/misavinov/jnd_datasets_val'
weights_path = '/home/misavinov/weights'
plots_path = '/home/misavinov/training_plots'

os.makedirs(weights_path, exist_ok=True)
os.makedirs(plots_path, exist_ok=True)

if __name__ == '__main__':
    is_jnd = args.layer != 'rgb'
    
    if args.layer == 'rgb':
        train_path, val_path = f'{data_dir}/train', f'{data_dir}/val'
    else:
        train_path = os.path.join(jnd_datasets_train, f'tinyimagenet_{args.layer}')
        val_path = os.path.join(jnd_datasets_val, f'tinyimagenet_{args.layer}')

    train_dataset = TinyImageNetDataset(train_path, is_jnd=is_jnd, is_train=True, layers=args.layer)
    val_dataset = TinyImageNetDataset(val_path, is_jnd=is_jnd, is_train=False, layers=args.layer)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    model = ResNet34().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    model_name = f'model_{args.layer}'
    loses_history, metric_history = train_eval(model, optimizer, criterion, scheduler, epochs, train_loader, val_loader, model_name, device)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    epochs_range = range(1, len(loses_history) + 1)
    
    ax1.plot(epochs_range, loses_history, color='blue', label='Train Loss')
    ax1.set_title('Train Loss')
    ax1.legend()
    
    ax2.plot(epochs_range, metric_history, color='red', label='Val acc')
    ax2.set_title('Val Acc')
    ax2.legend()
    
    plt.savefig(os.path.join(plots_path, model_name))
    plt.close()