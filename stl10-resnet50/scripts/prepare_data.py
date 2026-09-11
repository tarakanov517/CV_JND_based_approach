import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import os
from multiprocessing import Pool, freeze_support
from tqdm.auto import tqdm
import numpy as np
from pathlib import Path
from datasets import load_dataset

from utils.jnd_model import SimkinJNDModel
from utils.image_converter import ImageConverter

NUM_WORKERS = int(os.environ.get('SLURM_CPUS_PER_TASK', 4))

BASE_DIR = Path('/home/misavinov/scratch/ws/my_space/stl10-resnet50')
JND_DIR = BASE_DIR / 'stl10_xyzk'
L_MIN, L_MAX = 0.1, 300.0
TARGET_WIDTH_CM = [None, 4, 8, 16]

def init_worker():
    global converter, jnd_model
    converter = ImageConverter()
    jnd_model = SimkinJNDModel()

def process_single_image(task_args):
    idx, pil_image, target_width_cm = task_args

    try:
        rgb_image = pil_image.convert('RGB')
        np_image = np.array(rgb_image)
        xyzL_image = converter.RGB_to_xyzL(np_image)
        
        L_physical = xyzL_image[..., 3]

        jnd_model.find_La_with_background(L_patch = L_physical, target_width_cm=target_width_cm,)
        jnd_model.build_level_boundaries()
        k_map = jnd_model.L_to_k(L_physical)

        k_map_expanded = np.expand_dims(k_map, axis=-1)
        xyzk_matrix = np.concatenate([xyzL_image[..., :3], k_map_expanded], axis=-1).astype(np.float32)

        return idx, xyzk_matrix
        
    except Exception as e:
        print(f"Ошибка во время обработки idx {idx}: {e}")
        return idx, None
    
def generate_jnd_dataset(tasks, output_path, target_width_cm):

    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_samples = len(tasks)

    fp = np.lib.format.open_memmap(
        filename=output_path,
        mode="w+",
        dtype=np.float32,
        shape=(n_samples, 96, 96, 4),
    )

    args_list = [(t["idx"], t["image"], target_width_cm) for t in tasks]
    
    with Pool(processes=NUM_WORKERS, initializer=init_worker) as pool:
        for idx, matrix in tqdm(
            pool.imap_unordered(process_single_image, args_list, chunksize=16),
            total=n_samples,
            desc=output_path.name,
        ):
            fp[idx] = matrix

    fp.flush()
    del fp

if __name__ == '__main__':

    stl = load_dataset("jxie/stl10")

    tasks_train = [
        {
            "idx": idx,
            "image": item["image"],  # PIL.Image
        }
        for idx, item in enumerate(stl['train'])
    ]

    tasks_test = [
        {
            "idx": idx,
            "image": item["image"],  # PIL.Image
        }
        for idx, item in enumerate(stl['test'])
    ]

    freeze_support()
    for target_width_cm in TARGET_WIDTH_CM:
        print(f'Генерация для {target_width_cm}')
        generate_jnd_dataset(tasks_train, JND_DIR / str(target_width_cm) / 'train_jnd.npy', target_width_cm=target_width_cm)
        generate_jnd_dataset(tasks_test, JND_DIR / str(target_width_cm) / 'test_jnd.npy', target_width_cm=target_width_cm)
    print('Done!')