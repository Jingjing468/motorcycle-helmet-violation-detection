"""Safe path utilities used by validation and CLI scripts."""
from pathlib import Path


def ensure_directory(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def require_file(path, description="file"):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Required {description} not found: {path}")
    return path


def image_files(directory):
    supported = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(p for p in Path(directory).iterdir() if p.is_file() and p.suffix.lower() in supported)
