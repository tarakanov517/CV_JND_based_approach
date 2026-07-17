import os
import shutil
from multiprocessing import Pool
import glob
from tqdm.auto import tqdm
import numpy as np
from utils import ImageConverter, JNDModel

NUM_WORKERS = int(os.environ.get('SLURM_CPUS_PER_TASK', 4))
data_dir = '/scratch/misavinov/data'
dataset_folder = f'{data_dir}/tiny-imagenet-200'
val_dir = os.path.join(dataset_folder, 'val')
images_dir = os.path.join(val_dir, 'images')
annotations_file = os.path.join(val_dir, 'val_annotations.txt')

jnd_datasets_train = '/scratch/misavinov/jnd_datasets_train'
jnd_datasets_val = '/scratch/misavinov/jnd_datasets_val'

L_MIN, L_MAX = 0.1, 300.0

def sort_val_set():
    if os.path.exists(annotations_file) and os.path.exists(images_dir):
        with open(annotations_file, 'r') as f:
            lines = f.readlines()
        for line in tqdm(lines, desc="Сортировка val"):
            parts = line.strip().split('\t')
            img_name, cls_name = parts[0], parts[1]
            cls_folder = os.path.join(val_dir, cls_name, 'images')
            os.makedirs(cls_folder, exist_ok=True)
            src_path = os.path.join(images_dir, img_name)
            dst_path = os.path.join(cls_folder, img_name)
            if os.path.exists(src_path):
                shutil.move(src_path, dst_path)

def init_worker():
    global converter, jnd_model
    import math
    import numpy as np
    from scipy.stats import norm
    from scipy.optimize import minimize_scalar
    converter = ImageConverter()
    jnd_model = JNDModel(L_MIN, L_MAX)

def process_single_image(task):
    img_path, save_paths = task
    if all(os.path.exists(path) for path in save_paths.values()): return True
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

def save_jnd_folders(input_dir, base_output_dir):
    out_dirs = {k: os.path.join(base_output_dir, f'tinyimagenet_{k}') for k in ['kkk', 'xzk']}
    for d in out_dirs.values(): os.makedirs(d, exist_ok=True)
    
    tasks = []
    for cls_name in sorted(os.listdir(input_dir)):
        for d in out_dirs.values(): os.makedirs(os.path.join(d, cls_name), exist_ok=True)
        file_paths = glob.glob(os.path.join(input_dir, cls_name, 'images', "*.JPEG"))
        for path in file_paths:
            save_name = os.path.splitext(os.path.basename(path))[0] + ".npy"
            tasks.append((path, {k: os.path.join(out_dirs[k], cls_name, save_name) for k in out_dirs}))

    with Pool(processes=NUM_WORKERS, initializer=init_worker) as pool:
        with tqdm(total=len(tasks), desc=f"Создание датасетов {os.path.basename(base_output_dir)}") as pbar:
            for _ in pool.imap_unordered(process_single_image, tasks):
                pbar.update(1)

if __name__ == '__main__':
    sort_val_set()
    save_jnd_folders(f'{dataset_folder}/train', jnd_datasets_train)
    save_jnd_folders(f'{dataset_folder}/val', jnd_datasets_val)
    print("Генерация данных завершена!")