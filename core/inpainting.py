"""
core/inpainting.py
------------------
Rimozione oggetti tramite inpainting.

Backend
-------
  lama    — LaMa ONNX (Carve/LaMa-ONNX). Scaricato ~210MB al primo uso.
  patch   — Exemplar-based veloce: riempie un patch intero per iterazione
             invece di un pixel; usa cv2.matchTemplate (C++ / FFT).
  opencv  — OpenCV Telea. Istantaneo, ottimo per aree piccole.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Literal, Optional

import cv2
import numpy as np


# ── Cache ─────────────────────────────────────────────────────────────────────

def _cache_dir() -> Path:
    d = Path.home() / ".cache" / "phase11"
    d.mkdir(parents=True, exist_ok=True)
    return d


def lama_model_path() -> str | None:
    p = _cache_dir() / "lama.onnx"
    return str(p) if p.exists() else None


def is_lama_available() -> bool:
    return lama_model_path() is not None


# ── Download ──────────────────────────────────────────────────────────────────

_HF_REPO  = "Carve/LaMa-ONNX"
_HF_FILES = ["lama.onnx", "model.onnx", "LaMa.onnx", "lama_fp32.onnx"]


def download_lama(progress_cb: Optional[Callable[[float, str], None]] = None) -> str:
    dest = _cache_dir() / "lama.onnx"
    if dest.exists():
        return str(dest)
    if progress_cb:
        progress_cb(0.0, "Connessione a HuggingFace…")
    try:
        from huggingface_hub import hf_hub_download
        for fname in _HF_FILES:
            try:
                if progress_cb:
                    progress_cb(0.05, f"Tentativo {fname}…")
                path = hf_hub_download(
                    repo_id=_HF_REPO, filename=fname,
                    local_dir=str(_cache_dir()),
                    local_dir_use_symlinks=False,
                )
                src = Path(path)
                if src != dest:
                    src.rename(dest)
                if progress_cb:
                    progress_cb(1.0, "Modello scaricato.")
                reload_lama_session()
                return str(dest)
            except Exception:
                continue
        raise RuntimeError(f"Nessun file trovato in {_HF_REPO}")
    except ImportError:
        pass

    import urllib.request
    for fname in _HF_FILES:
        url = f"https://huggingface.co/{_HF_REPO}/resolve/main/{fname}"
        try:
            def _hook(count, block, total):
                if progress_cb and total > 0:
                    f = min(0.99, count * block / total)
                    progress_cb(f, f"Download… {f*100:.0f}% ({total/1e6:.0f} MB)")
            urllib.request.urlretrieve(url, str(dest), reporthook=_hook)
            if progress_cb:
                progress_cb(1.0, "Modello scaricato.")
            reload_lama_session()
            return str(dest)
        except Exception:
            dest.unlink(missing_ok=True)
            continue
    raise RuntimeError(f"Download fallito da {_HF_REPO}")


# ── LaMa ONNX runtime ─────────────────────────────────────────────────────────

_ort_session = None


def _get_ort_session():
    global _ort_session
    if _ort_session is not None:
        return _ort_session
    model = lama_model_path()
    if model is None:
        return None
    try:
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = max(1, os.cpu_count() or 4)
        _ort_session = ort.InferenceSession(
            model, sess_options=opts,
            providers=["CPUExecutionProvider"])
        return _ort_session
    except Exception:
        return None


def reload_lama_session() -> None:
    global _ort_session
    _ort_session = None


def _ceil_mod(x: int, mod: int) -> int:
    return ((x - 1) // mod + 1) * mod


def _lama_run_tile(sess, rgb_tile: np.ndarray, mask_tile: np.ndarray) -> np.ndarray:
    """
    Singola inferenza LaMa. Ridimensiona sempre alla dimensione fissa
    attesa dal modello (di solito 512×512), poi ritorna al size originale.
    """
    H, W = rgb_tile.shape[:2]

    # Leggi la dimensione attesa dal modello (può essere fissa o dinamica)
    expected_h = expected_w = _LAMA_SIZE
    for inp in sess.get_inputs():
        s = inp.shape
        if len(s) == 4:
            if isinstance(s[2], int) and s[2] > 0:
                expected_h = s[2]
            if isinstance(s[3], int) and s[3] > 0:
                expected_w = s[3]
            break

    # Ridimensiona a expected_h × expected_w
    img_r  = cv2.resize(rgb_tile,  (expected_w, expected_h),
                         interpolation=cv2.INTER_LANCZOS4)
    mask_r = cv2.resize(mask_tile, (expected_w, expected_h),
                         interpolation=cv2.INTER_NEAREST)

    mask_b = (mask_r > 127).astype(np.float32)
    img_t  = img_r.astype(np.float32).transpose(2, 0, 1)[np.newaxis] / 255.0
    mask_t = mask_b[np.newaxis, np.newaxis]

    inp_names = [i.name for i in sess.get_inputs()]
    out_name  = sess.get_outputs()[0].name
    out = sess.run([out_name], {inp_names[0]: img_t, inp_names[1]: mask_t})[0]
    raw = out[0].transpose(1, 2, 0)

    # Normalizzazione output — gestisce [0,1], [-1,1], [0,255]
    raw_min, raw_max = float(raw.min()), float(raw.max())
    if raw_min < -0.5:
        result_512 = ((raw + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    elif raw_max <= 1.1:
        result_512 = (raw * 255.0).clip(0, 255).astype(np.uint8)
    else:
        result_512 = raw.clip(0, 255).astype(np.uint8)

    # Ritorna al size originale del tile
    if (expected_h, expected_w) != (H, W):
        return cv2.resize(result_512, (W, H), interpolation=cv2.INTER_LANCZOS4)
    return result_512


def _lama_tiled_crop(sess, rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Tiling con overlap gaussiano sul crop."""
    H, W = rgb.shape[:2]
    step = max(_LAMA_SIZE // 2, 256)
    coords = np.linspace(-1, 1, _LAMA_SIZE)
    g = np.exp(-coords**2 * 2)
    win = np.outer(g, g).astype(np.float32)
    result  = np.zeros((H, W, 3), dtype=np.float32)
    weights = np.zeros((H, W),    dtype=np.float32)
    ys = list(range(0, max(1, H-_LAMA_SIZE+1), step))
    if not ys or ys[-1]+_LAMA_SIZE < H: ys.append(max(0, H-_LAMA_SIZE))
    xs = list(range(0, max(1, W-_LAMA_SIZE+1), step))
    if not xs or xs[-1]+_LAMA_SIZE < W: xs.append(max(0, W-_LAMA_SIZE))
    for y0 in ys:
        y1 = min(H, y0+_LAMA_SIZE)
        for x0 in xs:
            x1 = min(W, x0+_LAMA_SIZE)
            th, tw = y1-y0, x1-x0
            tile_r = _lama_run_tile(sess, rgb[y0:y1,x0:x1], mask[y0:y1,x0:x1])
            w = win[:th,:tw]
            result[y0:y1,x0:x1]  += tile_r.astype(np.float32) * w[:,:,np.newaxis]
            weights[y0:y1,x0:x1] += w
    return (result / np.maximum(weights[:,:,np.newaxis],1e-8)).clip(0,255).astype(np.uint8)


def _inpaint_lama(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    LaMa ONNX ottimizzato:
    - Crop sul bounding box della maschera + contesto proporzionale
    - Inferenza solo sul crop (non su tutta l'immagine)
    - Per RAW 4000x6000: da ~88 tile a ~4-9 tile
    - Output blendato con feathering sul bordo
    """
    sess = _get_ort_session()
    if sess is None:
        raise RuntimeError("Sessione LaMa non disponibile")
    H, W = rgb.shape[:2]
    hole = (mask > 127)
    ys, xs = np.where(hole)
    if len(ys) == 0:
        return rgb.copy()
    my0,my1 = int(ys.min()), int(ys.max())+1
    mx0,mx1 = int(xs.min()), int(xs.max())+1
    # Contesto: 50% della dimensione del buco, min 64px, max 256px
    ctx = int(np.clip(max(my1-my0, mx1-mx0) * 0.5, 64, 256))
    cy0 = max(0, my0-ctx); cy1 = min(H, my1+ctx)
    cx0 = max(0, mx0-ctx); cx1 = min(W, mx1+ctx)
    crop_rgb  = rgb[cy0:cy1, cx0:cx1].copy()
    crop_mask = mask[cy0:cy1, cx0:cx1].copy()
    ch, cw = crop_rgb.shape[:2]
    # Inferenza sul crop
    if ch <= _LAMA_SIZE and cw <= _LAMA_SIZE:
        result_crop = _lama_run_tile(sess, crop_rgb, crop_mask)
    else:
        result_crop = _lama_tiled_crop(sess, crop_rgb, crop_mask)
    # Paste diretto: copia il risultato del crop nella zona mascherata
    # Niente feathering che crea artefatti — paste pulito pixel per pixel
    result = rgb.copy()
    result[cy0:cy1, cx0:cx1] = result_crop
    return result


# ── Exemplar-based veloce ───────────────────────────────────────────────────── ─────────────────────────────────────────────────────

# Soglia pixel: sopra questa dimensione del buco si usa downscale automatico
_LAMA_SIZE = 512        # dimensione tile LaMa ONNX
_EXEMPLAR_MAX_PIXELS = 40_000  # ~200×200 px — oltre questo si scala


def _inpaint_exemplar(
    rgb:        np.ndarray,
    mask:       np.ndarray,
    patch_size: int = 9,
) -> np.ndarray:
    """
    Exemplar-based inpainting (Criminisi) ottimizzato.

    Per buchi fino a ~8000px (~90×90): gira a risoluzione piena.
    Per buchi più grandi: downscale automatico 2x/4x/8x finché il buco
    rientra nel limite, poi upscale del risultato con blend sul bordo.
    Questo mantiene i tempi sotto i 15-20 secondi anche per aree grandi.
    """
    hole_px = (mask > 127).sum()
    if hole_px == 0:
        return rgb.copy()

    # Calcola il fattore di scala necessario
    scale = 1
    while (hole_px // (scale*scale)) > _EXEMPLAR_MAX_PIXELS and scale < 16:
        scale *= 2

    if scale > 1:
        H, W = rgb.shape[:2]
        h_s, w_s = max(32, H // scale), max(32, W // scale)
        rgb_s  = cv2.resize(rgb,  (w_s, h_s), interpolation=cv2.INTER_AREA)
        mask_s = cv2.resize(mask, (w_s, h_s), interpolation=cv2.INTER_NEAREST)
        result_s = _exemplar_core(rgb_s, mask_s, patch_size)
        # Upscale del risultato
        result_up = cv2.resize(result_s, (W, H), interpolation=cv2.INTER_LANCZOS4)
        # Blend: usa il risultato upscalato solo nella zona mascherata
        # con un feathering di ~patch_size pixel sul bordo
        result = rgb.copy()
        hole_b = (mask > 127)
        # Paste diretto: nessun feathering che crea l'effetto "cristallo"
        # Il bordo naturale del patch è sufficiente per nascondere la giunzione
        result[hole_b] = result_up[hole_b]
        return result

    return _exemplar_core(rgb, mask, patch_size)


def _exemplar_core(
    rgb:        np.ndarray,
    mask:       np.ndarray,
    patch_size: int = 9,
) -> np.ndarray:
    """
    Cuore dell'exemplar-based inpainting. Usato su immagini già scalate.
    """
    H, W = rgb.shape[:2]
    patch_size = max(5, patch_size | 1)
    half = patch_size // 2

    img    = rgb.astype(np.float32)
    hole   = (mask > 127)
    filled = ~hole.copy()

    # Pre-inizializza la regione da riempire con Telea
    # → il matchTemplate trova subito patch decenti
    init = cv2.inpaint(rgb, (mask > 127).astype(np.uint8)*255, 3, cv2.INPAINT_TELEA)
    img[hole] = init[hole].astype(np.float32)

    kernel = np.ones((3,3), np.uint8)
    # Iterazioni massime: hole_pixels / (patch²/4) — molto meno del vecchio hole_pixels
    max_iters = max(10, (hole.sum() // max(1, (patch_size**2)//4)) + 10)

    for _ in range(max_iters):
        if filled.all():
            break

        boundary = (~filled).astype(np.uint8) & cv2.dilate(filled.astype(np.uint8), kernel)
        if boundary.sum() == 0:
            break

        # Priorità = confidenza × termine-dati (calcolati vettorialmente)
        filled_f  = filled.astype(np.float32)
        conf_map  = cv2.boxFilter(filled_f, -1, (patch_size, patch_size), normalize=True)
        gray_f    = cv2.cvtColor(np.clip(img,0,255).astype(np.uint8),
                                  cv2.COLOR_RGB2GRAY).astype(np.float32)
        gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
        nx = cv2.Sobel(filled_f, cv2.CV_32F, 1, 0, ksize=3)
        ny = cv2.Sobel(filled_f, cv2.CV_32F, 0, 1, ksize=3)
        data = (np.abs(gx*ny - gy*nx)
                / (np.sqrt(gx**2+gy**2)*np.sqrt(nx**2+ny**2) + 1e-8) + 0.001)
        priorities = conf_map * data * boundary.astype(np.float32)
        by, bx = np.unravel_index(np.argmax(priorities), priorities.shape)

        # Patch target
        ty0 = max(0, by-half); ty1 = min(H, by+half+1)
        tx0 = max(0, bx-half); tx1 = min(W, bx+half+1)
        ph, pw = ty1-ty0, tx1-tx0
        target = img[ty0:ty1, tx0:tx1]
        tmask  = filled[ty0:ty1, tx0:tx1].astype(np.float32)

        if tmask.sum() == 0:
            filled[by, bx] = True
            continue

        # Zona sorgente: patch completamente noti
        # erode con kernel del patch size = solo centri di patch pienamente noti
        src_ok = cv2.erode(filled.astype(np.uint8),
                            np.ones((ph, pw), np.uint8)) > 0

        # matchTemplate: SSD sui pixel noti del target
        t_masked = (target * tmask[:,:,np.newaxis]).astype(np.float32)
        best_sy, best_sx = ty0, tx0

        if ph == patch_size and pw == patch_size and src_ok.any():
            ssd = np.zeros((max(1,H-ph+1), max(1,W-pw+1)), dtype=np.float32)
            for c in range(3):
                ssd += cv2.matchTemplate(img[:,:,c], t_masked[:,:,c], cv2.TM_SQDIFF)
            sh, sw = ssd.shape
            # Escludi zone non completamente note
            ssd[~src_ok[:sh, :sw]] = float('inf')
            # Escludi patch destinazione stesso
            r0=max(0,ty0-half); r1=min(sh,ty0+half+1)
            c0=max(0,tx0-half); c1=min(sw,tx0+half+1)
            ssd[r0:r1, c0:c1] = float('inf')
            if not np.isinf(ssd).all():
                min_loc = cv2.minMaxLoc(ssd)[2]
                best_sx, best_sy = min_loc

        # Copia patch sorgente → destinazione (solo pixel non ancora riempiti)
        # Vettorizzato numpy — nessun loop Python
        unfilled_patch = ~filled[ty0:ty1, tx0:tx1]
        src_patch = img[best_sy:best_sy+ph, best_sx:best_sx+pw]
        if src_patch.shape == unfilled_patch.shape[::-1][::-1]:
            img[ty0:ty1, tx0:tx1][unfilled_patch] = src_patch[unfilled_patch]
            filled[ty0:ty1, tx0:tx1][unfilled_patch] = True

    return np.clip(img, 0, 255).astype(np.uint8)


# ── OpenCV Telea ──────────────────────────────────────────────────────────────

def _inpaint_opencv(rgb: np.ndarray, mask: np.ndarray, radius: int = 10) -> np.ndarray:
    return cv2.inpaint(rgb, (mask>127).astype(np.uint8)*255, radius, cv2.INPAINT_TELEA)


# ── API pubblica ──────────────────────────────────────────────────────────────

def inpaint(
    rgba:       np.ndarray,
    mask_bool:  np.ndarray,
    method:     Literal["lama", "patch", "opencv"] = "patch",
    patch_size: int = 9,
    radius:     int = 10,
) -> np.ndarray:
    """
    Rimuove l'area indicata dalla maschera e la riempie per inpainting.

    Parameters
    ----------
    rgba       : uint8 (H, W, 4)
    mask_bool  : bool  (H, W)  True = area da rimuovere
    method     : "lama"   → LaMa ONNX (massima qualità, richiede download)
                 "patch"  → Exemplar-based (buona qualità, nessun download)
                 "opencv" → OpenCV Telea (istantaneo, aree piccole)
    patch_size : per method="patch" (dispari, 5-15)
    radius     : per method="opencv"
    """
    rgb  = rgba[:, :, :3].copy()
    mask = mask_bool.astype(np.uint8) * 255

    if mask.max() == 0:
        return rgba.copy()

    if method == "lama":
        if is_lama_available():
            result_rgb = _inpaint_lama(rgb, mask)
        else:
            result_rgb = _inpaint_exemplar(rgb, mask, patch_size)
    elif method == "patch":
        result_rgb = _inpaint_exemplar(rgb, mask, patch_size)
    else:
        result_rgb = _inpaint_opencv(rgb, mask, radius)

    result = rgba.copy()
    result[:, :, :3] = result_rgb
    result[mask_bool, 3] = 255
    return result
