"""Дообучение ImageNet-VOneResNet50 на классификационном датасете (тот же рецепт, что фаза 2 late-simkin).

    python src/train_vone.py                        # параметры exp224.vone
    python src/train_vone.py dataset=imagewoof epochs=15
    python src/train_vone.py pretrained=null        # без ImageNet-претрейна (только для отладки)

Шум VOneBlock включён и при обучении, и при оценке (как в оригинальной статье).
"""
import sys
from pathlib import Path

import numpy as np
import torch
import rootutils
from omegaconf import OmegaConf

rootutils.setup_root(__file__, indicator="src", pythonpath=True)

from src.cls_data import get_loaders
from src.train_late_simkin import train_classifier
from src.vone_model import VOneClassifier, from_imagenet_ckpt, save_vone


def load_cfg(argv=None):
    root = OmegaConf.load("params.yaml")
    cli = OmegaConf.from_cli(list(argv if argv is not None else sys.argv[1:]))
    return OmegaConf.merge(root.exp224.vone, cli)


def run(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    
    print(OmegaConf.to_yaml(cfg))
    
    tr_loader, te_loader = get_loaders(cfg.dataset, cfg.batch_size, cfg.num_workers, cfg.data_root)
    if cfg.pretrained:
        model = from_imagenet_ckpt(cfg.pretrained, num_classes=10)
    else:
        print("[VOne] БЕЗ ImageNet-претрейна")
        model = VOneClassifier(num_classes=10)
    model = model.to(device)
    hist = train_classifier(model, tr_loader, te_loader, cfg.epochs, cfg.lr, device, amp=cfg.amp)
    out = Path(cfg.out_dir) / f"{cfg.dataset}_vone_s{cfg.seed}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    save_vone(model, out)
    print(f"saved {out} | clean={hist[-1]['test_acc']:.4f}")
    return out


if __name__ == "__main__":
    run(load_cfg())
