import os
import base64
from pathlib import Path
from PIL import Image, UnidentifiedImageError

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".heic"}


def collect_images(input_dir: str) -> list[Path]:
    root = Path(input_dir)
    paths = [
        p for p in root.rglob("*")
        if p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(paths)


def load_thumbnail(path: Path, max_size: int) -> Image.Image | None:
    try:
        img = Image.open(path).convert("RGB")
        img.thumbnail((max_size, max_size), Image.LANCZOS)
        return img
    except (UnidentifiedImageError, OSError):
        return None


def image_to_base64(path: Path, max_size: int) -> str | None:
    img = load_thumbnail(path, max_size)
    if img is None:
        return None
    import io
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.standard_b64encode(buf.getvalue()).decode()


def batch(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]
