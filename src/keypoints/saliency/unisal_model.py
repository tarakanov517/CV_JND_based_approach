import sys
import json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import zoom
import torch
from torchvision import transforms as T

from src.paths import UNISAL_DIR

def load_unisal_model(device: str | torch.device = "cpu"):
    """Загружает веса UNISAL из локальной папки unisal и переводит модель в eval."""
    if str(UNISAL_DIR) not in sys.path:
        sys.path.insert(0, str(UNISAL_DIR))

    from unisal.model import UNISAL

    cfg_path = UNISAL_DIR / "training_runs" / "pretrained_unisal" / "UNISAL.json"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    model = UNISAL(**cfg)
    model.load_weights(UNISAL_DIR / "training_runs" / "pretrained_unisal", "best")
    model.to(device).eval()
    return model

def predict_unisal_attention(
    image: np.ndarray | Image.Image | Path | str,
    model: torch.nn.Module,
    device: str | torch.device = "cpu",
    source: str = "SALICON",
    resize: bool = False,
    small_size: tuple[int, int] = (384, 224)
) -> np.ndarray:
    """Считает карту внимания UNISAL на оригинальном разрешении или с ресайзом."""
    if isinstance(image, (str, Path)):
        image_pil = Image.open(str(image)).convert("RGB")
    elif isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        image_pil = Image.fromarray(image)
    else:
        image_pil = image.convert("RGB")

    orig_w, orig_h = image_pil.size
    unisal_norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if resize:
        small_w, small_h = small_size
        img_input = image_pil.resize((small_w, small_h), Image.BILINEAR)
    else:
        full_h = (orig_h // 32) * 32
        full_w = (orig_w // 32) * 32
        img_input = image_pil.resize((full_w, full_h), Image.BILINEAR)

    tensor = unisal_norm(T.ToTensor()(img_input)).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(tensor, source=source)

    dens = np.exp(pred.squeeze().cpu().numpy())
    dens = zoom(dens, (orig_h / dens.shape[0], orig_w / dens.shape[1]), order=1)
    dens = (dens - dens.min()) / (dens.max() - dens.min() + 1e-8)
    return dens.astype(np.float64)
