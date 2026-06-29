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


# ──────────────────────────────────────────────────────────────────────────────
# Filtri spaziali: passa-alto / passa-basso / edge
# ──────────────────────────────────────────────────────────────────────────────

def gaussian_blur(img: np.ndarray, radius: int) -> np.ndarray:
    """Passa-basso gaussiano."""
    if radius <= 0:
        return img
    k = radius * 2 + 1
    out = img.copy()
    out[:, :, :3] = cv2.GaussianBlur(img[:, :, :3], (k, k), 0)
    return out


def median_blur(img: np.ndarray, radius: int) -> np.ndarray:
    """Passa-basso mediana — ottimo per rimuovere rumore sale-e-pepe."""
    if radius <= 0:
        return img
    k = radius * 2 + 1
    if k % 2 == 0:
        k += 1
    out = img.copy()
    out[:, :, :3] = cv2.medianBlur(img[:, :, :3], k)
    return out


def bilateral_blur(img: np.ndarray, radius: int, sigma: int = 50) -> np.ndarray:
    """Passa-basso bilaterale — preserva i bordi."""
    if radius <= 0:
        return img
    d = radius * 2 + 1
    out = img.copy()
    out[:, :, :3] = cv2.bilateralFilter(
        img[:, :, :3], d, float(sigma), float(sigma))
    return out


def unsharp_mask(img: np.ndarray, radius: int = 3, amount: float = 1.0) -> np.ndarray:
    """Unsharp mask: highpass aggiunto all'immagine. amount: 0.0–3.0."""
    if amount <= 0 or radius <= 0:
        return img
    k = radius * 2 + 1
    blurred = cv2.GaussianBlur(img[:, :, :3], (k, k), 0).astype(np.float32)
    sharp = img[:, :, :3].astype(np.float32)
    hp = sharp - blurred          # passa-alto
    result = sharp + hp * amount
    out = img.copy()
    out[:, :, :3] = _clip(result)
    return out


def high_boost(img: np.ndarray, amount: float = 1.5) -> np.ndarray:
    """High-boost filter: A*orig - blur. amount > 1 → boost dei dettagli."""
    if amount <= 0:
        return img
    blurred = cv2.GaussianBlur(img[:, :, :3], (5, 5), 0).astype(np.float32)
    orig = img[:, :, :3].astype(np.float32)
    result = orig * amount - blurred * (amount - 1)
    out = img.copy()
    out[:, :, :3] = _clip(result)
    return out


def laplacian(img: np.ndarray, amount: float = 1.0) -> np.ndarray:
    """Laplaciano: rileva i bordi e li sovrappone all'originale."""
    if amount <= 0:
        return img
    gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    lap_norm = np.clip((lap - lap.min()) / (lap.max() - lap.min() + 1e-8) * 255, 0, 255)
    lap3 = np.stack([lap_norm] * 3, axis=2).astype(np.float32)
    orig = img[:, :, :3].astype(np.float32)
    result = orig + lap3 * amount
    out = img.copy()
    out[:, :, :3] = _clip(result)
    return out


def emboss(img: np.ndarray, direction: int = 135, amount: float = 1.0) -> np.ndarray:
    """Effetto rilievo. direction: angolo 0-360°."""
    if amount <= 0:
        return img
    angle = np.radians(direction)
    dx = np.cos(angle)
    dy = np.sin(angle)
    kernel = np.array([
        [-dx - dy, -dy, dx - dy],
        [-dx,       0,       dx],
        [-dx + dy,  dy, dx + dy],
    ], dtype=np.float32) * amount
    gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2GRAY).astype(np.float32)
    em = cv2.filter2D(gray, -1, kernel) + 128
    em3 = np.stack([em, em, em], axis=2)
    out = img.copy()
    out[:, :, :3] = _clip(em3)
    return out


def sobel_edges(img: np.ndarray, amount: float = 1.0) -> np.ndarray:
    """Magnitude del gradiente Sobel sovrapposta all'originale."""
    if amount <= 0:
        return img
    gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    mag_norm = np.clip(mag / (mag.max() + 1e-8) * 255, 0, 255)
    mag3 = np.stack([mag_norm] * 3, axis=2)
    result = img[:, :, :3].astype(np.float32) + mag3 * amount
    out = img.copy()
    out[:, :, :3] = _clip(result)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Filtri in frequenza (FFT)
# ──────────────────────────────────────────────────────────────────────────────

def _fft_filter(
    img: np.ndarray,
    low_cut: float = 0.0,
    high_cut: float = 1.0,
) -> np.ndarray:
    """
    Filtro FFT generico. Applica una maschera frequenziale a disco
    nel dominio di Fourier per ogni canale RGB.
    
    low_cut  : raggio minimo (0.0–1.0, 0=DC) dei componenti conservati
    high_cut : raggio massimo (0.0–1.0, 1=Nyquist) dei componenti conservati
    Banda conservata = [low_cut, high_cut].
    """
    h, w = img.shape[:2]
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt(((X - cx) / (w / 2))**2 + ((Y - cy) / (h / 2))**2)

    mask = ((dist >= low_cut) & (dist <= high_cut)).astype(np.float32)
    # Smooth mask borders to reduce ringing
    k = max(1, min(h, w) // 50) * 2 + 1
    mask = cv2.GaussianBlur(mask, (k, k), k / 6.0)

    out = img.copy()
    for c in range(3):
        ch = img[:, :, c].astype(np.float32)
        F = np.fft.fft2(ch)
        Fs = np.fft.fftshift(F)
        Fs_filtered = Fs * mask
        F_back = np.fft.ifftshift(Fs_filtered)
        result = np.real(np.fft.ifft2(F_back))
        out[:, :, c] = _clip(result[:, :, np.newaxis]).squeeze()
    return out


def fft_lowpass(img: np.ndarray, cutoff: float = 0.3) -> np.ndarray:
    """
    Filtro passa-basso FFT. Rimuove dettagli fini e rumore.
    cutoff: frequenza di taglio 0.0–1.0 (0.1=solo basse freq, 0.9=quasi tutto)
    """
    return _fft_filter(img, low_cut=0.0, high_cut=max(0.01, cutoff))


def fft_highpass(img: np.ndarray, cutoff: float = 0.1) -> np.ndarray:
    """
    Filtro passa-alto FFT. Mantiene solo i dettagli fini e i bordi.
    cutoff: frequenza minima 0.0–1.0 (0.05=quasi tutto, 0.5=solo dettagli fini)
    """
    return _fft_filter(img, low_cut=max(0.0, cutoff), high_cut=1.0)


def fft_bandpass(
    img: np.ndarray,
    low: float = 0.1,
    high: float = 0.5,
) -> np.ndarray:
    """
    Filtro passa-banda FFT. Conserva una banda di frequenze [low, high].
    Utile per isolare texture a una scala specifica.
    """
    return _fft_filter(img, low_cut=max(0.0, low), high_cut=min(1.0, high))


def fft_remove_moire(img: np.ndarray, radius: float = 0.05) -> np.ndarray:
    """
    Rimozione moiré tramite FFT: azzera i picchi periodici ad alte frequenze.
    Individua i 10 picchi più forti fuori dal centro e li azzera.
    radius: raggio del notch attorno a ciascun picco (0.01–0.1).
    """
    h, w = img.shape[:2]
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt(((X - cx) / (w / 2))**2 + ((Y - cy) / (h / 2))**2)

    out = img.copy()
    for c in range(3):
        ch = img[:, :, c].astype(np.float32)
        F = np.fft.fft2(ch)
        Fs = np.fft.fftshift(F)
        mag = np.abs(Fs)

        # Trova picchi nell'area ad alta frequenza (>0.2 dal centro)
        high_freq_mask = dist_from_center > 0.2
        mag_masked = mag * high_freq_mask
        # Prendi i 10 picchi più forti
        flat = mag_masked.ravel()
        n_peaks = 10
        peak_indices = np.argpartition(flat, -n_peaks)[-n_peaks:]
        peak_ys, peak_xs = np.unravel_index(peak_indices, (h, w))

        # Crea notch mask (1 ovunque, 0 attorno ai picchi)
        notch_mask = np.ones((h, w), dtype=np.float32)
        notch_r_px = max(1, int(radius * min(h, w)))
        for py, px in zip(peak_ys, peak_xs):
            # Notch attorno al picco E al suo simmetrico (coniugato)
            for ny, nx in [(py, px), (h - 1 - py, w - 1 - px)]:
                ny, nx = int(ny), int(nx)
                y0 = max(0, ny - notch_r_px); y1 = min(h, ny + notch_r_px + 1)
                x0 = max(0, nx - notch_r_px); x1 = min(w, nx + notch_r_px + 1)
                notch_mask[y0:y1, x0:x1] = 0.0

        Fs_filtered = Fs * notch_mask
        F_back = np.fft.ifftshift(Fs_filtered)
        result = np.real(np.fft.ifft2(F_back))
        out[:, :, c] = np.clip(result, 0, 255).astype(np.uint8)

    return out


def fft_spectrum(img: np.ndarray) -> np.ndarray:
    """
    Restituisce la visualizzazione dello spettro FFT (ampiezza log-normalizzata)
    come immagine RGBA della stessa dimensione dell'input.
    Utile per analizzare la distribuzione delle frequenze.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2GRAY).astype(np.float32)
    F = np.fft.fft2(gray)
    Fs = np.fft.fftshift(F)
    mag = np.abs(Fs)
    mag_log = np.log1p(mag)
    mag_norm = (mag_log / (mag_log.max() + 1e-8) * 255).astype(np.uint8)
    spec = np.stack([mag_norm, mag_norm, mag_norm], axis=2)
    out = np.full((h, w, 4), 255, dtype=np.uint8)
    out[:, :, :3] = spec
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Applicazione su area selezionata
# ──────────────────────────────────────────────────────────────────────────────

def apply_to_selection(
    img:     np.ndarray,
    filtered: np.ndarray,
    sel_mask: np.ndarray,
) -> np.ndarray:
    """
    Combina immagine originale e immagine filtrata usando sel_mask.
    
    img      : RGBA uint8 originale (H,W,4)
    filtered : RGBA uint8 filtrata  (H,W,4) — stessa dimensione di img
    sel_mask : bool (H,W) — True = usa filtered, False = usa img originale
    
    Restituisce RGBA uint8 (H,W,4).
    """
    if sel_mask is None or not sel_mask.any():
        return filtered
    out = img.copy()
    # Feathering leggero sul bordo della selezione (2px)
    k = 5
    m = sel_mask.astype(np.float32)
    m_blur = cv2.GaussianBlur(m, (k, k), 1.5)[:, :, np.newaxis]
    out = out.astype(np.float32)
    filtered_f = filtered.astype(np.float32)
    # Blend solo sui canali RGB (alpha rimane dell'originale)
    out[:, :, :3] = (filtered_f[:, :, :3] * m_blur
                     + out[:, :, :3] * (1 - m_blur))
    return _clip(out)


# Registro filtri spaziali per il pannello UI
SPATIAL_FILTERS: dict[str, tuple[str, object, dict]] = {
    # key: (label, fn, default_params)
    "none":          ("Nessuno",             None,             {}),
    "gaussian":      ("Gaussiano",           gaussian_blur,    {"radius": 3}),
    "median":        ("Mediana",             median_blur,      {"radius": 2}),
    "bilateral":     ("Bilaterale",          bilateral_blur,   {"radius": 3, "sigma": 50}),
    "unsharp":       ("Unsharp Mask",        unsharp_mask,     {"radius": 3, "amount": 1.0}),
    "high_boost":    ("High-Boost",          high_boost,       {"amount": 1.5}),
    "laplacian":     ("Laplaciano",          laplacian,        {"amount": 1.0}),
    "emboss":        ("Rilievo (Emboss)",    emboss,           {"direction": 135, "amount": 1.0}),
    "sobel":         ("Bordi Sobel",         sobel_edges,      {"amount": 1.0}),
    "fft_low":       ("FFT Passa-basso",     fft_lowpass,      {"cutoff": 0.3}),
    "fft_high":      ("FFT Passa-alto",      fft_highpass,     {"cutoff": 0.1}),
    "fft_band":      ("FFT Passa-banda",     fft_bandpass,     {"low": 0.1, "high": 0.5}),
    "fft_moire":     ("FFT Rimuovi Moiré",  fft_remove_moire, {"radius": 0.05}),
    "fft_spectrum":  ("FFT Spettro",         fft_spectrum,     {}),
}
