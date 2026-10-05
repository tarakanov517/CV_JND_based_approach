from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

GROUPS = {
    "natural": ["contrast", "brightness", "fog"],
    "digital": ["jpeg_compression", "pixelate", "elastic_transform"],
    "noise": ["gaussian_noise", "shot_noise", "impulse_noise"],
}
ALL = [c for g in GROUPS.values() for c in g]


def group_of(corruption: str) -> str:
    return next((g for g, cs in GROUPS.items() if corruption in cs), "other")


def _ensure_pkg_resources():
    """imagecorruptions импортирует pkg_resources, которого нет в свежем setuptools -> минимальный шим."""
    try:
        import pkg_resources  # noqa: F401
    except ImportError:
        import importlib.util
        import os
        import sys
        import types

        def resource_filename(pkg, name):
            origin = importlib.util.find_spec(pkg).origin
            return os.path.normpath(os.path.join(os.path.dirname(origin), name))

        shim = types.ModuleType("pkg_resources")
        shim.resource_filename = resource_filename
        sys.modules["pkg_resources"] = shim


def _import_corrupt():
    """imagecorruptions 1.1.x не обновлялся под numpy 2 / skimage >= 0.19: чиним на лету."""
    _ensure_pkg_resources()
    if not hasattr(np, "float_"):                     # fog: np.float_ удалён в numpy 2.0
        np.float_ = np.float64
    import imagecorruptions.corruptions as icc
    if not getattr(icc.gaussian, "_patched", False):  # glass/gaussian_blur: multichannel -> channel_axis
        orig = icc.gaussian

        def gaussian(*a, multichannel=None, **kw):
            if multichannel is not None and "channel_axis" not in kw:
                kw["channel_axis"] = -1 if multichannel else None
            return orig(*a, **kw)
        gaussian._patched = True
        icc.gaussian = gaussian
    from imagecorruptions import corrupt
    return corrupt


def _corrupt_chunk(images, idx0, corruption, severity):
    corrupt = _import_corrupt()
    out = np.empty_like(images)
    for j, img in enumerate(images):
        np.random.seed(idx0 + j)                     # детерминированные шумовые искажения
        o = corrupt(img, corruption_name=corruption, severity=severity)
        out[j] = np.clip(np.asarray(o, dtype=np.float32), 0, 255).round().astype(np.uint8)
    return out


def corrupted_images(images: np.ndarray, corruption: str, severity: int,
                     cache_path: Path | None = None,
                     n_jobs: int = -1, chunk: int = 64) -> np.ndarray:
    """images: (N,H,W,3) uint8. Если cache_path задан — кешируется в .npy."""
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            return np.load(cache_path)

    parts = Parallel(n_jobs=n_jobs)(
        delayed(_corrupt_chunk)(images[i:i + chunk], i, corruption, severity)
        for i in range(0, len(images), chunk))
    out = np.concatenate(parts)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, out)

    return out
