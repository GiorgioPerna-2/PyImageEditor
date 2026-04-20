"""
core/raw_io.py
--------------
Lettura file RAW fotografici (.cr2, .nef, .arw, .dng, ecc.)
tramite rawpy (libraw) — tutto in locale, nessun cloud.

Dipendenza opzionale: rawpy  (pip install rawpy)
Se non installato, solleva ImportError con messaggio chiaro.

Restituisce sempre un array RGBA uint8 (H, W, 4).
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

# Estensioni RAW — copertura equivalente a GIMP (LibRaw) e Photoshop (Camera Raw)
RAW_EXTENSIONS = {
    # Canon
    ".cr2", ".cr3", ".crw", ".ciff",
    # Nikon
    ".nef", ".nrw",
    # Sony
    ".arw", ".srf", ".sr2",
    # Adobe
    ".dng",
    # Fujifilm
    ".raf",
    # Panasonic / Lumix
    ".rw2", ".raw",
    # Olympus / OM System
    ".orf",
    # Pentax / Ricoh
    ".pef", ".ptx", ".dng",
    # Sigma
    ".x3f",
    # Leica
    ".rwl", ".lrw", ".raw",
    # Hasselblad
    ".3fr", ".fff",
    # Phase One / Leaf
    ".iiq", ".cap", ".eip",
    # Mamiya
    ".mef", ".mfw",
    # Minolta / Konica-Minolta
    ".mrw", ".mdc",
    # Kodak
    ".dcr", ".kdc", ".k25", ".dcs",
    # Epson
    ".erf",
    # Casio
    ".bay",
    # Samsung
    ".srw",
    # Blackmagic
    ".braw",   # rawpy supporta braw tramite plugin
    # Generico
    ".raw",
}


def is_raw(path: str | Path) -> bool:
    return Path(path).suffix.lower() in RAW_EXTENSIONS


def load_raw(
    path: str | Path,
    use_camera_wb: bool = True,
    half_size: bool = False,
    output_bps: int = 8,
) -> np.ndarray:
    """
    Carica un file RAW e restituisce un array RGBA uint8.

    Parameters
    ----------
    path           : percorso al file RAW
    use_camera_wb  : usa il bilanciamento del bianco registrato dalla fotocamera
    half_size      : dimezza la risoluzione (più veloce)
    output_bps     : bit per sample output (8 o 16; se 16 → normalizzato a uint8)

    Returns
    -------
    RGBA uint8 (H, W, 4)  — alpha sempre 255
    """
    try:
        import rawpy
    except ImportError:
        raise ImportError(
            "rawpy non è installato.\n"
            "Installa con:  pip install rawpy\n"
            "oppure aggiungi rawpy a pyproject.toml."
        )

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File non trovato: {path}")

    with rawpy.imread(str(path)) as raw:
        rgb = raw.postprocess(
            use_camera_wb=use_camera_wb,
            half_size=half_size,
            output_bps=output_bps,
            no_auto_bright=False,
        )

    # rawpy restituisce RGB uint8 (o uint16 se output_bps=16)
    if rgb.dtype != np.uint8:
        # Normalizza 16-bit → 8-bit
        rgb = (rgb / 256).astype(np.uint8)

    # Aggiungi canale alpha pieno
    alpha = np.full((*rgb.shape[:2], 1), 255, dtype=np.uint8)
    rgba = np.concatenate([rgb, alpha], axis=2)
    return rgba
