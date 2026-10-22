# python src/train_vone.py                              # параметры exp224.vone
# python src/train_vone.py dataset=imagewoof epochs=15
# python src/train_vone.py noise_scales=[0.70,0.35,0.20,0.10]
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
    """Дообучить VOne под каждый noise_scale из cfg; вернуть пути чекпойнтов."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(OmegaConf.to_yaml(cfg))
    tr_loader, te_loader = get_loaders(cfg.dataset, cfg.batch_size, cfg.num_workers, cfg.data_root)

    scales = list(cfg.noise_scales) if cfg.get("noise_scales") else [cfg.get("noise_scale", None)]
    outs = []
    for ns in scales:
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)       # одинаковый старт для всех точек
        
        if cfg.pretrained:
            model = from_imagenet_ckpt(cfg.pretrained, num_classes=10)
        else:
            print("[VOne] без ImageNet-претрейна")
            model = VOneClassifier(num_classes=10)

        tag = ""
        if ns is not None:                                          # перебить дефолтный шум из флагов ckpt
            model.arch["noise_scale"] = float(ns)
            model.arch["noise_level"] = float(cfg.get("noise_level", model.arch["noise_level"]))
            model.net.vone_block.set_noise_mode(model.arch["noise_mode"],
                                                model.arch["noise_scale"], model.arch["noise_level"])
            tag = f"_sc{model.arch['noise_scale']:g}"
            print(f"[VOne] noise_scale={model.arch['noise_scale']} level={model.arch['noise_level']}")

        model = model.to(device)
        hist = train_classifier(model, tr_loader, te_loader, cfg.epochs, cfg.lr, device, amp=cfg.amp)
        
        out = Path(cfg.out_dir) / f"{cfg.dataset}_vone{tag}_s{cfg.seed}.pt"
        out.parent.mkdir(parents=True, exist_ok=True)
        
        save_vone(model, out)
        
        print(f"saved {out} | clean={hist[-1]['test_acc']:.4f}")
        outs.append(out)
        
        del model
        torch.cuda.empty_cache()
    return outs


if __name__ == "__main__":
    run(load_cfg())