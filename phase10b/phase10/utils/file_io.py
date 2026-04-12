from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

# Formati supportati in lettura e scrittura
SUPPORTED_READ = "Immagini (*.png *.jpg *.jpeg *.webp *.tiff *.tif *.bmp *.gif)"
SUPPORTED_WRITE = "PNG (*.png);;JPEG (*.jpg *.jpeg);;WebP (*.webp);;TIFF (*.tiff *.tif);;BMP (*.bmp)"


def load_image(path: str | Path) -> np.ndarray:
    """
    Carica un'immagine da disco e la restituisce come array numpy RGBA uint8.
    Converte sempre in RGBA per uniformità interna.
    """
    img = Image.open(path).convert("RGBA")
    return np.asarray(img, dtype=np.uint8)


def save_image(array: np.ndarray, path: str | Path, quality: int = 95) -> None:
    """
    Salva un array numpy (RGBA uint8) su disco.
    Il formato viene dedotto dall'estensione.
    Per JPEG converte automaticamente in RGB (JPEG non supporta alpha).
    """
    path = Path(path)
    img = Image.fromarray(array, mode="RGBA")
    ext = path.suffix.lower()

    if ext in (".jpg", ".jpeg"):
        # JPEG non supporta trasparenza
        img = img.convert("RGB")
        img.save(path, format="JPEG", quality=quality, optimize=True)
    elif ext == ".webp":
        img.save(path, format="WEBP", quality=quality, lossless=False)
    elif ext in (".tiff", ".tif"):
        img.save(path, format="TIFF")
    else:
        # PNG e BMP mantengono RGBA
        img.save(path)


def array_to_pil(array: np.ndarray) -> Image.Image:
    """Converte np.ndarray RGBA → PIL Image."""
    return Image.fromarray(array, mode="RGBA")


def pil_to_array(img: Image.Image) -> np.ndarray:
    """Converte PIL Image → np.ndarray RGBA uint8."""
    return np.asarray(img.convert("RGBA"), dtype=np.uint8)
