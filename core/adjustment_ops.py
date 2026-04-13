"""
core/adjustment_ops.py
----------------------
Operazioni di regolazione non-distruttive applicate durante il compositing.
Input/output: np.ndarray float32 shape (H, W, 4), valori 0–255.
Tutti i calcoli avvengono su float32; il clip finale è a carico del chiamante.
"""
from __future__ import annotations
import numpy as np


def apply_adjustment(kind: str, params: dict, img: np.ndarray) -> np.ndarray:
    """Dispatcher: chiama la funzione giusta in base al kind."""
    fns = {
        "levels":        _levels,
        "hsl":           _hsl,
        "color_balance": _color_balance,
        "curves":        _curves,
        "exposure":      _exposure,
    }
    fn = fns.get(kind)
    if fn is None:
        return img.copy()
    return fn(params, img)


# ── Livelli ───────────────────────────────────────────────────────────────────

def _levels(params: dict, img: np.ndarray) -> np.ndarray:
    """
    Livelli input/output + gamma su ciascun canale RGB.
    params:
      in_black:  0–255 (default 0)
      in_white:  0–255 (default 255)
      gamma:     0.1–10.0 (default 1.0)
      out_black: 0–255 (default 0)
      out_white: 0–255 (default 255)
    """
    in_b  = float(params.get("in_black",  0))
    in_w  = float(params.get("in_white",  255))
    gamma = float(params.get("gamma",     1.0))
    out_b = float(params.get("out_black", 0))
    out_w = float(params.get("out_white", 255))

    out = img.copy()
    rgb = out[:, :, :3]
    # Normalizza in [0,1] rispetto all'input range
    rgb = np.clip((rgb - in_b) / max(in_w - in_b, 1e-6), 0.0, 1.0)
    # Applica gamma
    if abs(gamma - 1.0) > 1e-4:
        rgb = np.power(rgb, 1.0 / max(gamma, 1e-4))
    # Rimappa all'output range
    out[:, :, :3] = rgb * (out_w - out_b) + out_b
    return out


# ── HSL ───────────────────────────────────────────────────────────────────────

def _hsl(params: dict, img: np.ndarray) -> np.ndarray:
    """
    Regolazione Tonalità / Saturazione / Luminosità.
    params:
      hue:        -180 – +180 gradi (default 0)
      saturation: -100 – +100 (default 0)
      lightness:  -100 – +100 (default 0)
    """
    import cv2
    hue_shift = float(params.get("hue", 0))
    sat_adj   = float(params.get("saturation", 0))
    lig_adj   = float(params.get("lightness",  0))

    out = img.copy()
    rgb = np.clip(out[:, :, :3], 0, 255).astype(np.uint8)
    hls = cv2.cvtColor(rgb, cv2.COLOR_RGB2HLS).astype(np.float32)

    # H in OpenCV: 0–180
    hls[:, :, 0] = (hls[:, :, 0] + hue_shift / 2.0) % 180.0
    # L: 0–255
    hls[:, :, 1] = np.clip(hls[:, :, 1] + lig_adj * 2.55, 0, 255)
    # S: 0–255
    hls[:, :, 2] = np.clip(hls[:, :, 2] * (1 + sat_adj / 100.0), 0, 255)

    rgb2 = cv2.cvtColor(hls.astype(np.uint8), cv2.COLOR_HLS2RGB).astype(np.float32)
    out[:, :, :3] = rgb2
    return out


# ── Bilanciamento colore ──────────────────────────────────────────────────────

def _color_balance(params: dict, img: np.ndarray) -> np.ndarray:
    """
    Bilanciamento colore shadows / midtones / highlights.
    params per ciascuna fascia (shadows | midtones | highlights):
      {fascia}_cyan_red:     -100 – +100
      {fascia}_magenta_green:-100 – +100
      {fascia}_yellow_blue:  -100 – +100
    """
    out = img.copy()
    rgb = out[:, :, :3] / 255.0   # [0,1]

    for band, lo, hi in [("shadows", 0.0, 0.33),
                          ("midtones", 0.25, 0.75),
                          ("highlights", 0.67, 1.0)]:
        cr  = float(params.get(f"{band}_cyan_red",      0)) / 100.0
        mg  = float(params.get(f"{band}_magenta_green", 0)) / 100.0
        yb  = float(params.get(f"{band}_yellow_blue",   0)) / 100.0

        if abs(cr) + abs(mg) + abs(yb) < 1e-4:
            continue

        lum = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
        # Peso gaussiano sulla fascia di luminosità
        mid = (lo + hi) / 2.0
        sig = (hi - lo) / 2.0
        weight = np.exp(-0.5 * ((lum - mid) / max(sig, 1e-4)) ** 2)[:, :, np.newaxis]

        rgb = rgb + weight * np.array([cr, mg, yb])[np.newaxis, np.newaxis, :]

    out[:, :, :3] = np.clip(rgb * 255.0, 0, 255)
    return out


# ── Curves (semplificato: un punto di controllo per canale) ───────────────────

def _curves(params: dict, img: np.ndarray) -> np.ndarray:
    """
    Curva tonale con punto di controllo centrale per ogni canale.
    params:
      master_in / master_out: 0–255 (punto centrale della curva master)
      r_in / r_out, g_in / g_out, b_in / b_out: idem per i singoli canali
    """
    out = img.copy()

    def _apply_channel(ch_data: np.ndarray, in_pt: float, out_pt: float) -> np.ndarray:
        """Curva spline quadratica passante per (0,0), (in_pt, out_pt), (255,255)."""
        x = np.clip(ch_data, 0, 255) / 255.0
        t = in_pt / 255.0
        s = out_pt / 255.0
        # Bézier quadratica: B(x) = (1-x/t)^2*0 + 2*(x/t)*(1-x/t)*s + (x/t)^2*1
        # Semplificato come interpolazione lineare a tratti
        result = np.where(
            x <= t,
            x * (s / max(t, 1e-6)),
            s + (x - t) * ((1 - s) / max(1 - t, 1e-6))
        )
        return np.clip(result * 255.0, 0, 255)

    # Master
    m_in  = float(params.get("master_in",  128))
    m_out = float(params.get("master_out", 128))
    if abs(m_in - 128) > 1 or abs(m_out - 128) > 1:
        for c in range(3):
            out[:, :, c] = _apply_channel(out[:, :, c], m_in, m_out)

    # Canali individuali
    for c, key in enumerate(("r", "g", "b")):
        ci  = float(params.get(f"{key}_in",  128))
        co  = float(params.get(f"{key}_out", 128))
        if abs(ci - 128) > 1 or abs(co - 128) > 1:
            out[:, :, c] = _apply_channel(out[:, :, c], ci, co)

    return out


# ── Esposizione ───────────────────────────────────────────────────────────────

def _exposure(params: dict, img: np.ndarray) -> np.ndarray:
    """
    Esposizione / Offset / Gamma (stile Lightroom).
    params:
      exposure: -5.0 – +5.0 stop (default 0)
      offset:   -0.5 – +0.5 (default 0)
      gamma:    0.1 – 10.0 (default 1.0)
    """
    exposure = float(params.get("exposure", 0.0))
    offset   = float(params.get("offset",   0.0))
    gamma    = float(params.get("gamma",    1.0))

    out = img.copy()
    rgb = out[:, :, :3] / 255.0
    # Esposizione: moltiplicazione in spazio lineare
    rgb = rgb * (2 ** exposure)
    # Offset
    rgb = rgb + offset
    # Gamma
    rgb = np.clip(rgb, 0, 1)
    if abs(gamma - 1.0) > 1e-4:
        rgb = np.power(rgb, 1.0 / max(gamma, 1e-4))
    out[:, :, :3] = np.clip(rgb * 255.0, 0, 255)
    return out
