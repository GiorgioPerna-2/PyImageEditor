"""
core/upscale.py
---------------
Upscaling con modelli ESRGAN pre-trainati, con fallback a bicubic.

Il modello viene scaricato automaticamente la prima volta (≈ 64 MB per RealESRGAN_x4).
Successivamente viene letto dalla cache locale (~/.cache/image_suite/models/).

Dipendenze opzionali:
  - basicsr / realesrgan  (pip install realesrgan)
  - torch                 (richiesto da realesrgan)
Se non installate, usa OpenCV bicubic come fallback.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

MODELS_CACHE = Path.home() / ".cache" / "image_suite" / "models"

# Modelli supportati: chiave → (nome file peso, scala)
ESRGAN_MODELS: dict[str, tuple[str, int]] = {
    "RealESRGAN x4": ("RealESRGAN_x4plus.pth", 4),
    "RealESRGAN x4 (anime)": ("RealESRGAN_x4plus_anime_6B.pth", 4),
    "RealESRGAN x2": ("RealESRGAN_x2plus.pth", 2),
}

_URL_BASE = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/"
_URLS: dict[str, str] = {
    "RealESRGAN_x4plus.pth":         _URL_BASE + "RealESRGAN_x4plus.pth",
    "RealESRGAN_x4plus_anime_6B.pth": _URL_BASE + "RealESRGAN_x4plus_anime_6B.pth",
    "RealESRGAN_x2plus.pth":         _URL_BASE + "RealESRGAN_x2plus.pth",
}


def _bicubic_fallback(rgba: np.ndarray, scale: int) -> np.ndarray:
    import cv2
    h, w = rgba.shape[:2]
    return cv2.resize(rgba, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)


def available_models() -> list[str]:
    return list(ESRGAN_MODELS.keys())


def model_is_cached(model_name: str) -> bool:
    if model_name not in ESRGAN_MODELS:
        return False
    fname, _ = ESRGAN_MODELS[model_name]
    return (MODELS_CACHE / fname).exists()


def upscale(
    rgba: np.ndarray,
    model_name: str,
    progress_cb=None,          # callable(str) per aggiornamenti di stato
) -> np.ndarray:
    """
    Esegue l'upscaling di `rgba` (RGBA uint8) con il modello indicato.
    Se realesrgan/torch non sono disponibili, usa bicubic.
    `progress_cb` viene chiamata con messaggi di testo (opzionale).
    """
    if model_name not in ESRGAN_MODELS:
        raise ValueError(f"Modello sconosciuto: {model_name!r}")

    fname, scale = ESRGAN_MODELS[model_name]

    def _msg(s: str) -> None:
        if progress_cb:
            progress_cb(s)

    # ── Prova realesrgan ──────────────────────────────────────────────
    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
        import torch

        model_path = MODELS_CACHE / fname
        if not model_path.exists():
            _msg(f"Scarico {fname} …")
            _download(fname, model_path)

        _msg("Carico il modello…")
        # Parametri secondo il modello
        if "anime_6B" in fname:
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                            num_block=6, num_grow_ch=32, scale=scale)
        else:
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                            num_block=23, num_grow_ch=32, scale=scale)

        upsampler = RealESRGANer(
            scale=scale,
            model_path=str(model_path),
            model=model,
            tile=512,          # elabora a tile per limitare la VRAM
            tile_pad=10,
            pre_pad=0,
            half=False,
        )

        _msg("Upscaling in corso…")
        # realesrgan lavora su BGR
        import cv2
        bgr = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)
        out_bgr, _ = upsampler.enhance(bgr, outscale=scale)
        out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)

        # Scala anche il canale alpha con bicubic
        h, w = rgba.shape[:2]
        alpha_up = cv2.resize(rgba[:, :, 3], (w * scale, h * scale),
                              interpolation=cv2.INTER_LINEAR)
        result = np.dstack([out_rgb, alpha_up])
        _msg("Completato.")
        return result.astype(np.uint8)

    except ImportError:
        _msg("realesrgan/torch non disponibili — uso bicubic.")
        result = _bicubic_fallback(rgba, scale)
        _msg("Completato (bicubic).")
        return result
    except Exception as exc:
        _msg(f"Errore: {exc} — uso bicubic.")
        return _bicubic_fallback(rgba, scale)


def _download(fname: str, dest: Path) -> None:
    import urllib.request
    MODELS_CACHE.mkdir(parents=True, exist_ok=True)
    url = _URLS.get(fname)
    if not url:
        raise ValueError(f"URL non trovato per {fname}")
    urllib.request.urlretrieve(url, dest)
