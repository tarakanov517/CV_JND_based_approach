import shutil
from pathlib import Path


def make_dir(path: str | Path, delete_if_exist: bool = False) -> Path:
    """Создать директорию (опционально пересоздав её с нуля)."""
    path = Path(path)
    if delete_if_exist and path.is_dir():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path