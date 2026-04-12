from __future__ import annotations

import numpy as np
import cv2


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _clip(img: np.ndarray) -> np.ndarray:
    return np.clip(img, 0, 255).astype(np.uint8)


def _blend(original: np.ndarray, modified: np.ndarray, intensity: int) -> np.ndarray:
    """Miscela lineare tra immagine originale e modificata. intensity: 0-100."""
    t = max(0, min(100, intensity)) / 100.0
    out = original.copy().astype(np.float32)
    out[:, :, :3] = (1 - t) * original[:, :, :3] + t * modified[:, :, :3].astype(np.float32)
    return _clip(out)


# ──────────────────────────────────────────────────────────────────────────────
# Regolazioni tonali
# ──────────────────────────────────────────────────────────────────────────────

def brightness(img: np.ndarray, val: int) -> np.ndarray:
    out = img.astype(np.float32)
    out[:, :, :3] += val
    return _clip(out)


def contrast(img: np.ndarray, val: int) -> np.ndarray:
    factor = (259 * (val + 255)) / (255 * (259 - val))
    out = img.astype(np.float32)
    out[:, :, :3] = factor * (out[:, :, :3] - 128) + 128
    return _clip(out)


def saturation(img: np.ndarray, val: int) -> np.ndarray:
    rgb = img[:, :, :3]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1 + val / 100.0), 0, 255)
    rgb2 = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    out = img.copy()
    out[:, :, :3] = rgb2
    return out


def sharpen(img: np.ndarray, amount: float) -> np.ndarray:
    if amount <= 0:
        return img
    kernel = np.array([[0, -1, 0], [-1, 5 + amount, -1], [0, -1, 0]])
    rgb = img[:, :, :3]
    sharp = cv2.filter2D(rgb, -1, kernel)
    out = img.copy()
    out[:, :, :3] = sharp
    return _clip(out)


def blur(img: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return img
    k = radius * 2 + 1
    rgb = cv2.GaussianBlur(img[:, :, :3], (k, k), 0)
    out = img.copy()
    out[:, :, :3] = rgb
    return out


def hue_shift(img: np.ndarray, degrees: int) -> np.ndarray:
    if degrees == 0:
        return img
    rgb = img[:, :, :3]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + degrees / 2.0) % 180
    rgb2 = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)
    out = img.copy()
    out[:, :, :3] = rgb2
    return out


def vibrance(img: np.ndarray, amount: int) -> np.ndarray:
    if amount == 0:
        return img
    rgb = img[:, :, :3]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    sat = hsv[:, :, 1] / 255.0
    boost = (1 - sat) * (amount / 100.0) * 128
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] + boost, 0, 255)
    rgb2 = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    out = img.copy()
    out[:, :, :3] = rgb2
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Stili colore
# ──────────────────────────────────────────────────────────────────────────────

def grayscale(img: np.ndarray, intensity: int = 100) -> np.ndarray:
    rgb = img[:, :, :3].astype(np.float32)
    lum = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
    gray3 = np.stack([lum, lum, lum], axis=2)
    bw_img = img.copy()
    bw_img[:, :, :3] = _clip(gray3)
    return _blend(img, bw_img, intensity)


def sepia(img: np.ndarray, intensity: int = 100) -> np.ndarray:
    rgb = img[:, :, :3].astype(np.float32)
    r = rgb[:, :, 0] * 0.393 + rgb[:, :, 1] * 0.769 + rgb[:, :, 2] * 0.189
    g = rgb[:, :, 0] * 0.349 + rgb[:, :, 1] * 0.686 + rgb[:, :, 2] * 0.168
    b = rgb[:, :, 0] * 0.272 + rgb[:, :, 1] * 0.534 + rgb[:, :, 2] * 0.131
    sep_img = img.copy()
    sep_img[:, :, :3] = _clip(np.stack([r, g, b], axis=2))
    return _blend(img, sep_img, intensity)


def warm(img: np.ndarray, intensity: int = 100) -> np.ndarray:
    t = intensity / 100.0
    out = img.copy().astype(np.float32)
    out[:, :, 0] = np.clip(out[:, :, 0] + 30 * t, 0, 255)
    out[:, :, 1] = np.clip(out[:, :, 1] + 12 * t, 0, 255)
    out[:, :, 2] = np.clip(out[:, :, 2] - 22 * t, 0, 255)
    return _clip(out)


def cool(img: np.ndarray, intensity: int = 100) -> np.ndarray:
    t = intensity / 100.0
    out = img.copy().astype(np.float32)
    out[:, :, 0] = np.clip(out[:, :, 0] - 22 * t, 0, 255)
    out[:, :, 1] = np.clip(out[:, :, 1] + 5 * t, 0, 255)
    out[:, :, 2] = np.clip(out[:, :, 2] + 30 * t, 0, 255)
    return _clip(out)


def vintage(img: np.ndarray, intensity: int = 100) -> np.ndarray:
    t = intensity / 100.0
    out = img.copy().astype(np.float32)
    out[:, :, :3] = out[:, :, :3] * (1 - 0.15 * t) + 25 * t
    out[:, :, 0] = np.clip(out[:, :, 0] + 18 * t, 0, 255)
    out[:, :, 2] = np.clip(out[:, :, 2] - 18 * t, 0, 255)
    return _clip(out)


def fade(img: np.ndarray, intensity: int = 100) -> np.ndarray:
    t = intensity / 100.0
    out = img.copy().astype(np.float32)
    out[:, :, :3] = out[:, :, :3] * (1 - 0.28 * t) + 35 * t
    return _clip(out)


def cross_process(img: np.ndarray, intensity: int = 100) -> np.ndarray:
    t = intensity / 100.0
    out = img.copy().astype(np.float32)
    r = out[:, :, 0] / 255.0
    g = out[:, :, 1] / 255.0
    b = out[:, :, 2] / 255.0
    out[:, :, 0] = np.clip((r + t * r * (1 - r) * 0.8) * 255, 0, 255)
    out[:, :, 1] = np.clip((g - t * g * (1 - g) * 0.4) * 255, 0, 255)
    out[:, :, 2] = np.clip((b + t * (1 - b) * b * 0.6) * 255, 0, 255)
    return _clip(out)


def invert(img: np.ndarray, intensity: int = 100) -> np.ndarray:
    inv = img.copy()
    inv[:, :, :3] = 255 - img[:, :, :3]
    return _blend(img, inv, intensity)


def duotone(img: np.ndarray, intensity: int = 100) -> np.ndarray:
    """Duotone viola/arancio."""
    rgb = img[:, :, :3].astype(np.float32)
    lum = (0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]) / 255.0
    shadow = np.array([80.0, 20.0, 160.0])
    highlight = np.array([255.0, 160.0, 30.0])
    duo = shadow[None, None, :] * (1 - lum[:, :, None]) + highlight[None, None, :] * lum[:, :, None]
    duo_img = img.copy()
    duo_img[:, :, :3] = _clip(duo)
    return _blend(img, duo_img, intensity)


def cyanotype(img: np.ndarray, intensity: int = 100) -> np.ndarray:
    """Cianotipia: stampa fotografica storica in toni di blu."""
    rgb = img[:, :, :3].astype(np.float32)
    lum = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
    cyan = np.stack([lum * 0.35, lum * 0.65, lum * 0.9], axis=2)
    cyan_img = img.copy()
    cyan_img[:, :, :3] = _clip(cyan)
    return _blend(img, cyan_img, intensity)


# Registro stili disponibili per il pannello UI
COLOR_STYLES: dict[str, tuple[str, object]] = {
    "none":          ("Nessuno",       None),
    "bw":            ("Bianco e Nero", grayscale),
    "sepia":         ("Seppia",        sepia),
    "warm":          ("Caldo",         warm),
    "cool":          ("Freddo",        cool),
    "vintage":       ("Vintage",       vintage),
    "fade":          ("Fade",          fade),
    "cross_process": ("Cross-process", cross_process),
    "invert":        ("Invertito",     invert),
    "duotone":       ("Duotone",       duotone),
    "cyanotype":     ("Cianotipia",    cyanotype),
}


def apply_color_style(img: np.ndarray, style_key: str, intensity: int) -> np.ndarray:
    entry = COLOR_STYLES.get(style_key)
    if entry is None or entry[1] is None:
        return img
    return entry[1](img, intensity)  # type: ignore[operator]


# ──────────────────────────────────────────────────────────────────────────────
# Effetti fotografici
# ──────────────────────────────────────────────────────────────────────────────

def vignette(img: np.ndarray, strength: int) -> np.ndarray:
    if strength <= 0:
        return img
    h, w = img.shape[:2]
    t = strength / 100.0
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt(((X - w / 2) / (w / 2)) ** 2 + ((Y - h / 2) / (h / 2)) ** 2)
    mask = np.clip(1 - dist * t * 0.85, 0, 1)
    out = img.copy().astype(np.float32)
    out[:, :, :3] *= mask[:, :, np.newaxis]
    return _clip(out)


def film_grain(img: np.ndarray, amount: int) -> np.ndarray:
    """Grana filmica. Seed fisso per preview stabile."""
    if amount <= 0:
        return img
    rng = np.random.default_rng(seed=42)
    noise = rng.normal(0, amount * 0.8, img.shape[:2]).astype(np.float32)
    out = img.copy().astype(np.float32)
    out[:, :, :3] += noise[:, :, np.newaxis]
    return _clip(out)


def soft_glow(img: np.ndarray, amount: int) -> np.ndarray:
    """Bagliore soft: blend Screen con versione sfocat."""
    if amount <= 0:
        return img
    t = amount / 100.0
    rgb = img[:, :, :3]
    blurred = cv2.GaussianBlur(rgb, (51, 51), 0).astype(np.float32)
    base = rgb.astype(np.float32)
    screen = 255 - ((255 - base) * (255 - blurred) / 255)
    out = img.copy()
    out[:, :, :3] = _clip(base * (1 - t) + screen * t)
    return out


def clarity(img: np.ndarray, amount: int) -> np.ndarray:
    """Microcontrasto locale (medie frequenze), come in Lightroom."""
    if amount == 0:
        return img
    t = amount / 100.0
    rgb = img[:, :, :3].astype(np.float32)
    blur_light = cv2.GaussianBlur(rgb, (5, 5), 0).astype(np.float32)
    blur_heavy = cv2.GaussianBlur(rgb, (31, 31), 0).astype(np.float32)
    mid_freq = blur_light - blur_heavy
    out = img.copy()
    out[:, :, :3] = _clip(rgb + mid_freq * t * 2)
    return out


def dehaze(img: np.ndarray, amount: int) -> np.ndarray:
    """Riduce la foschia atmosferica (dark channel prior semplificato)."""
    if amount <= 0:
        return img
    t = amount / 100.0
    out = img.copy().astype(np.float32)
    dark = np.min(out[:, :, :3], axis=2)
    dark_blur = cv2.GaussianBlur(dark, (15, 15), 0)[:, :, np.newaxis]
    out[:, :, :3] = np.clip(
        (out[:, :, :3] - dark_blur * t * 0.5 - 10 * t) * (1 + t * 0.3),
        0, 255
    )
    return _clip(out)
