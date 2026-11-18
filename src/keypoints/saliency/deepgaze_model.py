from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import zoom
from scipy.special import logsumexp
import torch
import deepgaze_pytorch

from src.paths import CENTERBIAS_PATH

def load_deepgaze_iie_model(device: str | torch.device = "cpu"):
    """Загружает предобученную модель DeepGaze IIE."""
    model = deepgaze_pytorch.DeepGazeIIE(pretrained=True).to(device).eval()
    return model

def predict_deepgaze_iie(
    image: np.ndarray | Image.Image | Path | str,
    model: torch.nn.Module,
    cb_template: np.ndarray | None = None,
    device: str | torch.device = "cpu"
) -> np.ndarray:
    """Считает карту внимания DeepGaze IIE на оригинальном разрешении изображения."""
    if isinstance(image, (str, Path)):
        image_pil = Image.open(str(image)).convert("RGB")
    elif isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        image_pil = Image.fromarray(image)
    else:
        image_pil = image.convert("RGB")

    image_np = np.array(image_pil)
    h, w = image_np.shape[:2]

    if cb_template is None:
        cb_template = np.load(str(CENTERBIAS_PATH))

    cb = zoom(cb_template, (h / cb_template.shape[0], w / cb_template.shape[1]), order=0, mode="nearest")
    cb -= logsumexp(cb)
    cb_tensor = torch.from_numpy(cb).unsqueeze(0).float().to(device)

    tensor_img = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).float().to(device)

    with torch.no_grad():
        pred = model(tensor_img, cb_tensor)

    dens = np.exp(pred.squeeze().cpu().numpy())
    dens = (dens - dens.min()) / (dens.max() - dens.min() + 1e-8)
    return dens.astype(np.float64)
