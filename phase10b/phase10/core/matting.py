"""
core/matting.py
---------------
Strumenti di scontornamento: rembg (automatico) e GrabCut (interattivo).

Entrambe le funzioni lavorano su array RGBA uint8 (H, W, 4) e restituiscono
un nuovo array RGBA con lo sfondo reso trasparente.
"""
from __future__ import annotations

import numpy as np
import cv2


# ── rembg — rimozione sfondo automatica ──────────────────────────────────────

def remove_background_rembg(
    rgba: np.ndarray,
    model_name: str = "u2net",
    alpha_matting: bool = False,
    alpha_matting_fg_threshold: int = 240,
    alpha_matting_bg_threshold: int = 10,
    alpha_matting_erode_size: int = 10,
) -> np.ndarray:
    """
    Rimuove lo sfondo usando rembg (U2-Net o modelli alternativi).

    Parametri:
    - model_name: "u2net" (generale), "u2net_human_seg" (ritratti),
                  "isnet-general-use" (qualità superiore, più lento),
                  "silueta" (veloce, ritratti)
    - alpha_matting: se True, usa alpha matting per bordi più morbidi
                     (più lento ma risultati migliori su capelli/pelo)
    """
    from rembg import remove, new_session
    session = new_session(model_name)

    from PIL import Image
    pil_in = Image.fromarray(rgba, "RGBA")
    pil_out = remove(
        pil_in,
        session=session,
        alpha_matting=alpha_matting,
        alpha_matting_foreground_threshold=alpha_matting_fg_threshold,
        alpha_matting_background_threshold=alpha_matting_bg_threshold,
        alpha_matting_erode_size=alpha_matting_erode_size,
    )
    return np.asarray(pil_out.convert("RGBA"), dtype=np.uint8)


# ── GrabCut — scontornamento interattivo ─────────────────────────────────────

def grabcut_rect(
    rgba: np.ndarray,
    rect: tuple[int, int, int, int],   # (x, y, w, h) in pixel immagine
    iterations: int = 5,
) -> np.ndarray:
    """
    Segmenta il soggetto dentro 'rect' usando l'algoritmo GrabCut di OpenCV.

    Il rettangolo deve contenere il soggetto principale; tutto ciò che è
    fuori dal rettangolo viene considerato sfondo.

    Restituisce un RGBA con lo sfondo trasparente.
    """
    h, w = rgba.shape[:2]
    x, y, rw, rh = rect
    x  = max(0, x);  y  = max(0, y)
    rw = min(rw, w - x); rh = min(rh, h - y)
    if rw <= 0 or rh <= 0:
        return rgba.copy()

    # GrabCut opera su BGR
    bgr = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)

    mask = np.zeros((h, w), dtype=np.uint8)
    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)

    cv2.grabCut(bgr, mask, (x, y, rw, rh),
                bgd_model, fgd_model, iterations,
                cv2.GC_INIT_WITH_RECT)

    # mask: 0=BG, 1=FG, 2=probable BG, 3=probable FG
    fg_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
                       255, 0).astype(np.uint8)

    # Applica maschera sull'alpha
    out = rgba.copy()
    out[:, :, 3] = (out[:, :, 3].astype(np.float32) *
                    fg_mask.astype(np.float32) / 255.0).astype(np.uint8)
    return out


def refine_mask_morphology(
    rgba: np.ndarray,
    erode_px: int = 1,
    dilate_px: int = 2,
    blur_px: int = 3,
) -> np.ndarray:
    """
    Post-processing della maschera alpha: erode i bordi frastagliati,
    poi dilata leggermente e sfuma per transizioni morbide.
    Utile dopo GrabCut.
    """
    out = rgba.copy()
    alpha = out[:, :, 3]

    if erode_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (erode_px * 2 + 1, erode_px * 2 + 1))
        alpha = cv2.erode(alpha, kernel, iterations=1)

    if dilate_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1))
        alpha = cv2.dilate(alpha, kernel, iterations=1)

    if blur_px > 0:
        k = blur_px * 2 + 1
        alpha = cv2.GaussianBlur(alpha, (k, k), 0)

    out[:, :, 3] = alpha
    return out
