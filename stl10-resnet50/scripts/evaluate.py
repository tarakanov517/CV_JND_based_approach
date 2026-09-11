import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import os
import json
import torch
import torchattacks
from tqdm.auto import tqdm

from utils.jnd_model import SimkinJNDModel
from utils.image_converter import ImageConverter
from utils.auxiliary_functions import make_backbone, get_loader01
from utils.bpda import BPDAModelWrapper

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = int(os.environ.get('SLURM_CPUS_PER_TASK', 4))
L_MIN, L_MAX = 0.1, 300.0

EPS_LIST = [2/255, 4/255, 8/255, 16/255]

BASE_DIR = Path('/home/misavinov/scratch/ws/my_space/stl10-resnet50')
weights_path = BASE_DIR / 'weights'
evaluate_plots_path = BASE_DIR / 'evaluate_plots'
results_dir = BASE_DIR / 'results'

os.makedirs(results_dir, exist_ok=True)
os.makedirs(evaluate_plots_path, exist_ok=True)

MODELS_CONFIG = [
    {
        "name": "model_rgb",
        "layer": "rgb",
        "target_width_cm": None,
        "weights": weights_path / "model_rgb.pth"
    },
    {
        "name": "model_kkk_None",
        "layer": "kkk",
        "target_width_cm": None,
        "weights": weights_path / "model_kkk_None.pth"
    },
    {
        "name": "model_kkk_4cm",
        "layer": "kkk",
        "target_width_cm": 4.0,
        "weights": weights_path / "model_kkk_4cm.pth"
    },
    {
        "name": "model_kkk_8cm",
        "layer": "kkk",
        "target_width_cm": 8.0,
        "weights": weights_path / "model_kkk_8cm.pth"
    },
    {
        "name": "model_kkk_16cm",
        "layer": "kkk",
        "target_width_cm": 16.0,
        "weights": weights_path / "model_kkk_16cm.pth"
    },
    {
        "name": "model_xzk_None",
        "layer": "xzk",
        "target_width_cm": None,
        "weights": weights_path / "model_xzk_None.pth"
    },
    {
        "name": "model_xzk_4cm",
        "layer": "xzk",
        "target_width_cm": 4.0,
        "weights": weights_path / "model_xzk_4cm.pth"
    },
    {
        "name": "model_xzk_8cm",
        "layer": "xzk",
        "target_width_cm": 8.0,
        "weights": weights_path / "model_xzk_8cm.pth"
    },
    {
        "name": "model_xzk_16cm",
        "layer": "xzk",
        "target_width_cm": 16.0,
        "weights": weights_path / "model_xzk_16cm.pth"
    },
]

RGB_MODEL_CONFIG = MODELS_CONFIG[0]

def compute_metrics_dict(diff_tensor):
    diff_flat = diff_tensor.reshape(diff_tensor.size(0), -1)
    l1 = torch.norm(diff_flat, p=1, dim=1).cpu().tolist()
    l2 = torch.norm(diff_flat, p=2, dim=1).cpu().tolist()
    linf = torch.max(torch.abs(diff_flat), dim=1)[0].cpu().tolist()
    return l1, l2, linf

def get_target_features(wrapper, x_tensor):
    with torch.no_grad():
        if hasattr(wrapper, 'transform'):
            return wrapper.transform(x_tensor)
        return None

if __name__ == '__main__':
    converter = ImageConverter()
    jnd_model = SimkinJNDModel()

    test_loader_1000 = get_loader01(n=1000, batch_size=128, num_workers=NUM_WORKERS)

    rgb_model = make_backbone().to(device)
    rgb_model.load_state_dict(torch.load(RGB_MODEL_CONFIG['weights'], map_location=device))
    rgb_model.eval()

    target_models = {}
    for config in MODELS_CONFIG:
        target_model = make_backbone().to(device)
        target_model.load_state_dict(torch.load(config['weights'], map_location=device))
        target_model.eval()

        target_models[config['name']] = BPDAModelWrapper(
            classifier_model=target_model,
            converter=converter,
            jnd_model=jnd_model,
            layer_type=config['layer'],
            target_width_cm=config['target_width_cm']
        )

    clean_predictions = {cfg['name']: [] for cfg in MODELS_CONFIG}
    all_targets = []

    with torch.no_grad():
        for x, y in tqdm(test_loader_1000, desc='Clean evaluation', leave=False):
            x = x.to(device)
            all_targets.append(y.cpu())
            for cfg in MODELS_CONFIG:
                name = cfg['name']
                p_clean = target_models[name](x).argmax(dim=1).cpu()
                clean_predictions[name].append(p_clean)

    all_targets = torch.cat(all_targets)
    for name in clean_predictions:
        clean_predictions[name] = torch.cat(clean_predictions[name])

    results = {'transfer': {}}

    for eps in tqdm(EPS_LIST, desc='Transfer eps', leave=True):
        results['transfer'][eps] = {}

        attack = torchattacks.PGD(
            rgb_model,
            eps=eps,
            alpha=eps * 2.5 / 20,
            steps=20
        )

        adv_predictions = {cfg['name']: [] for cfg in MODELS_CONFIG}
        norms_storage = {
            cfg['name']: {
                'rgb': {'l1': [], 'l2': [], 'linf': []},
                'target': {'l1': [], 'l2': [], 'linf': []}
            } for cfg in MODELS_CONFIG
        }

        for x, y in tqdm(test_loader_1000, desc=f'Batch (eps={eps:.4f})', leave=False):
            x = x.to(device)
            y = y.to(device)
            x_adv = attack(x, y)

            diff_rgb = x_adv - x
            b_l1, b_l2, b_linf = compute_metrics_dict(diff_rgb)

            for cfg in MODELS_CONFIG:
                name = cfg['name']
                wrapper = target_models[name]
                p_adv = wrapper(x_adv).argmax(dim=1).cpu()
                adv_predictions[name].append(p_adv)

                norms_storage[name]['rgb']['l1'].extend(b_l1)
                norms_storage[name]['rgb']['l2'].extend(b_l2)
                norms_storage[name]['rgb']['linf'].extend(b_linf)

                z_clean = get_target_features(wrapper, x)
                z_adv = get_target_features(wrapper, x_adv)
                if z_clean is not None and z_adv is not None:
                    t_l1, t_l2, t_linf = compute_metrics_dict(z_adv - z_clean)
                    norms_storage[name]['target']['l1'].extend(t_l1)
                    norms_storage[name]['target']['l2'].extend(t_l2)
                    norms_storage[name]['target']['linf'].extend(t_linf)

        total_samples = all_targets.size(0)
        for cfg in MODELS_CONFIG:
            name = cfg['name']
            p_clean = clean_predictions[name]
            p_adv = torch.cat(adv_predictions[name])

            clean_mask = (p_clean == all_targets)
            robust_mask = (p_adv == all_targets)
            fooled_mask = clean_mask & (~robust_mask)

            clean_cnt = clean_mask.sum().item()
            robust_cnt = robust_mask.sum().item()
            fooled_cnt = fooled_mask.sum().item()

            results['transfer'][eps][name] = {
                'clean_acc': clean_cnt / total_samples,
                'robust_acc': robust_cnt / total_samples,
                'asr': fooled_cnt / clean_cnt if clean_cnt > 0 else 0.0,
                'norms_rgb': norms_storage[name]['rgb'],
                'norms_target': norms_storage[name]['target']
            }

    results['bpda'] = {}

    for config in MODELS_CONFIG:
        name = config['name']
        results['bpda'][name] = {}
        wrapper = target_models[name]

        for eps in tqdm(EPS_LIST, desc=f"BPDA eps ({name})", leave=True):
            torch.cuda.empty_cache()

            attack = torchattacks.PGD(
                wrapper,
                eps=eps,
                alpha=eps * 2.5 / 20,
                steps=20
            )

            adv_preds_list = []
            rgb_l1, rgb_l2, rgb_linf = [], [], []
            tgt_l1, tgt_l2, tgt_linf = [], [], []

            for x, y in tqdm(test_loader_1000, desc='Batch', leave=False):
                x = x.to(device)
                y = y.to(device)

                x_adv = attack(x, y)

                diff_rgb = x_adv - x
                b_l1, b_l2, b_linf = compute_metrics_dict(diff_rgb)
                rgb_l1.extend(b_l1)
                rgb_l2.extend(b_l2)
                rgb_linf.extend(b_linf)

                z_clean = get_target_features(wrapper, x)
                z_adv = get_target_features(wrapper, x_adv)
                if z_clean is not None and z_adv is not None:
                    t_l1, t_l2, t_linf = compute_metrics_dict(z_adv - z_clean)
                    tgt_l1.extend(t_l1)
                    tgt_l2.extend(t_l2)
                    tgt_linf.extend(t_linf)

                with torch.no_grad():
                    preds_adv = wrapper(x_adv).argmax(dim=1).cpu()
                    adv_preds_list.append(preds_adv)

            p_clean = clean_predictions[name]
            p_adv = torch.cat(adv_preds_list)

            clean_mask = (p_clean == all_targets)
            robust_mask = (p_adv == all_targets)
            fooled_mask = clean_mask & (~robust_mask)

            clean_cnt = clean_mask.sum().item()
            robust_cnt = robust_mask.sum().item()
            fooled_cnt = fooled_mask.sum().item()

            results['bpda'][name][eps] = {
                'clean_acc': clean_cnt / total_samples,
                'robust_acc': robust_cnt / total_samples,
                'asr': fooled_cnt / clean_cnt if clean_cnt > 0 else 0.0,
                'norms_rgb': {
                    'l1': rgb_l1,
                    'l2': rgb_l2,
                    'linf': rgb_linf
                },
                'norms_target': {
                    'l1': tgt_l1,
                    'l2': tgt_l2,
                    'linf': tgt_linf
                }
            }

    output_json_path = results_dir / 'attack_results.json'

    formatted_results = {
        'transfer': {},
        'bpda': {}
    }

    for eps, models in results['transfer'].items():
        eps_key = f"{int(round(eps * 255))}/255"
        formatted_results['transfer'][eps_key] = models

    for model_name, eps_data in results['bpda'].items():
        formatted_results['bpda'][model_name] = {}
        for eps, metrics in eps_data.items():
            eps_key = f"{int(round(eps * 255))}/255"
            formatted_results['bpda'][model_name][eps_key] = metrics

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(formatted_results, f, indent=4)