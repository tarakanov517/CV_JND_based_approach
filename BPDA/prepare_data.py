import os
from multiprocessing import Pool, freeze_support
import glob
from tqdm.auto import tqdm
import numpy as np
from pathlib import Path

from utils import (
    JNDModel,
    ImageConverter
)

NUM_WORKERS = int(os.environ.get('SLURM_CPUS_PER_TASK', 4))

BASE_DIR = Path('/home/misavinov/scratch/BPDA')
DATA_DIR = BASE_DIR / 'data'
JND_DIR = DATA_DIR / 'cifar10_val_jnd'
L_MIN, L_MAX = 0.1, 300.0

def init_worker():
    global converter, jnd_model
    converter = ImageConverter()
    jnd_model = JNDModel(L_MIN, L_MAX)

def process_single_image(task):
    img_path, save_paths = task

    if all(os.path.exists(p) for p in save_paths.values()):
        return True
    
    try:
        V = converter.read_img(img_path)
        norm_img = converter.normalize_image(V)
        linRGBimg = converter.Inverse_sRGB_Companding(norm_img)
        XYZ_image = converter.Linear_RGB_to_XYZ(linRGBimg)

        xyL_image = converter.XYZ_to_xyY(XYZ_image, L_MAX)
        xzL_image = converter.XYZ_to_xzL(XYZ_image, L_MAX)

        x = xyL_image[..., 0]
        z = xzL_image[..., 1]
        
        L_physical = L_MIN + (L_MAX - L_MIN) * xyL_image[..., 2]

        jnd_model.find_La(L_physical)
        jnd_model.build_level_boundaries()
        k_map = jnd_model.L_to_k(L_physical)

        if 'kkk' in save_paths and not os.path.exists(save_paths['kkk']):
            np.save(save_paths['kkk'], np.stack([k_map, k_map, k_map], axis=-1).astype(np.float32))
            
        if 'xzk' in save_paths and not os.path.exists(save_paths['xzk']):
            np.save(save_paths['xzk'], np.stack([x, z, k_map], axis=-1).astype(np.float32))

        return True
        
    except Exception as e:
        print(f"Ошибка во время обработки {img_path}: {e}")
        return False
    
def save_jnd_folder(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    out_dirs = {k: output_dir / f"cifar10_{k}" for k in ['kkk', 'xzk']}
    for d in out_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        print(f"Папка {input_dir} не найдена!")
        return

    classes = sorted(os.listdir(input_dir))
    tasks = []
    
    for cls_name in classes:
        cls_folder = input_dir / cls_name
        if not cls_folder.is_dir():
            continue

        for d in out_dirs.values():
            (d / cls_name).mkdir(parents=True, exist_ok=True)

        file_paths = glob.glob(str(cls_folder / "*.png")) + glob.glob(str(cls_folder / "*.jpg"))
        
        for path in file_paths:
            file_name = Path(path).stem + ".npy"
            save_paths = {k: str(out_dirs[k] / cls_name / file_name) for k in out_dirs}
            tasks.append((path, save_paths))
    
    with Pool(processes=NUM_WORKERS, initializer=init_worker) as pool:
        with tqdm(total=len(tasks), desc=f"Создание {output_dir.name}") as pbar:
            for _ in pool.imap_unordered(process_single_image, tasks):
                pbar.update(1)

if __name__ == '__main__':
    freeze_support()
    save_jnd_folder(DATA_DIR / 'test', JND_DIR / 'test_jnd')
    print('Done!')