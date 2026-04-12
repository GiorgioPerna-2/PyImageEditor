"""
core/selection.py
-----------------
Maschera di selezione e operazioni associate.

La maschera è un array bool (H, W) nello spazio del canvas composito.
True = pixel selezionato. None = nessuna selezione attiva.
"""
from __future__ import annotations

import math
import numpy as np
import cv2
from typing import Optional


# ── Costruzione maschera ──────────────────────────────────────────────────────

def mask_from_rect(canvas_h: int, canvas_w: int,
                   x: int, y: int, w: int, h: int) -> np.ndarray:
    """Maschera rettangolare."""
    m = np.zeros((canvas_h, canvas_w), dtype=bool)
    x2 = min(canvas_w, x + w)
    y2 = min(canvas_h, y + h)
    x  = max(0, x);  y = max(0, y)
    if x2 > x and y2 > y:
        m[y:y2, x:x2] = True
    return m


def mask_from_ellipse(canvas_h: int, canvas_w: int,
                      x: int, y: int, w: int, h: int) -> np.ndarray:
    """Maschera ellittica iscritta nel rettangolo (x,y,w,h)."""
    m = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    cx = x + w // 2;  cy = y + h // 2
    cv2.ellipse(m, (cx, cy), (max(1, w // 2), max(1, h // 2)),
                0, 0, 360, 1, thickness=-1)
    return m.astype(bool)


def mask_from_lasso(canvas_h: int, canvas_w: int,
                    points: list[tuple[int, int]]) -> np.ndarray:
    """Maschera da poligono lasso (lista di punti image-coords)."""
    if len(points) < 3:
        return np.zeros((canvas_h, canvas_w), dtype=bool)
    m = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(m, [pts], 1)
    return m.astype(bool)


# ── Operazioni ────────────────────────────────────────────────────────────────

def feather_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    """
    Ammorbidisce i bordi della selezione con un blur gaussiano.
    Restituisce una maschera float32 [0,1].
    """
    if radius <= 0:
        return mask.astype(np.float32)
    k = radius * 2 + 1
    m = mask.astype(np.float32)
    return cv2.GaussianBlur(m, (k, k), radius / 2.0)


def invert_mask(mask: np.ndarray) -> np.ndarray:
    return ~mask


def expand_mask(mask: np.ndarray, px: int) -> np.ndarray:
    """Espande la selezione di px pixel."""
    if px <= 0:
        return mask
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (px * 2 + 1, px * 2 + 1))
    return cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)


def shrink_mask(mask: np.ndarray, px: int) -> np.ndarray:
    """Restringe la selezione di px pixel."""
    if px <= 0:
        return mask
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (px * 2 + 1, px * 2 + 1))
    return cv2.erode(mask.astype(np.uint8), kernel).astype(bool)


def select_all(canvas_h: int, canvas_w: int) -> np.ndarray:
    return np.ones((canvas_h, canvas_w), dtype=bool)


# ── Operazioni su layer ───────────────────────────────────────────────────────

def copy_region(layer_data: np.ndarray,
                mask: np.ndarray,
                layer_offset_x: int = 0,
                layer_offset_y: int = 0) -> np.ndarray:
    """
    Ritaglia la regione del layer selezionata dalla maschera canvas.
    Restituisce un RGBA con i pixel fuori maschera trasparenti,
    ritagliato al bounding box della selezione.
    """
    h, w = layer_data.shape[:2]
    ch, cw = mask.shape

    # Bounding box della selezione sul canvas
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return np.zeros((1, 1, 4), dtype=np.uint8)
    y1, y2 = int(np.argmax(rows)), int(len(rows) - np.argmax(rows[::-1]))
    x1, x2 = int(np.argmax(cols)), int(len(cols) - np.argmax(cols[::-1]))

    # Canvas coords → layer coords
    ly1 = y1 - layer_offset_y;  ly2 = y2 - layer_offset_y
    lx1 = x1 - layer_offset_x;  lx2 = x2 - layer_offset_x

    # Regione di intersezione
    lly1 = max(0, ly1);  lly2 = min(h, ly2)
    llx1 = max(0, lx1);  llx2 = min(w, lx2)
    patch_h = y2 - y1;   patch_w = x2 - x1

    out = np.zeros((patch_h, patch_w, 4), dtype=np.uint8)
    if lly2 > lly1 and llx2 > llx1:
        dst_y1 = lly1 - ly1;  dst_x1 = llx1 - lx1
        dst_y2 = dst_y1 + (lly2 - lly1)
        dst_x2 = dst_x1 + (llx2 - llx1)
        out[dst_y1:dst_y2, dst_x1:dst_x2] = layer_data[lly1:lly2, llx1:llx2]

    # Applica la maschera di selezione
    sel_patch = mask[y1:y2, x1:x2].astype(np.float32)
    out[:, :, 3] = (out[:, :, 3].astype(np.float32) * sel_patch).astype(np.uint8)
    return out


def cut_region(layer_data: np.ndarray,
               mask: np.ndarray,
               layer_offset_x: int = 0,
               layer_offset_y: int = 0) -> np.ndarray:
    """
    Azzera l'alpha nella regione selezionata del layer (taglia).
    Modifica in-place e restituisce il layer modificato.
    """
    out = layer_data.copy()
    h, w = out.shape[:2]
    ch, cw = mask.shape

    for cy in range(ch):
        ly = cy - layer_offset_y
        if ly < 0 or ly >= h:
            continue
        for cx_start in range(cw):
            if not mask[cy, cx_start]:
                continue
            # Trova il run di True
            cx_end = cx_start
            while cx_end < cw and mask[cy, cx_end]:
                cx_end += 1
            lx1 = max(0, cx_start - layer_offset_x)
            lx2 = min(w, cx_end - layer_offset_x)
            if lx2 > lx1:
                out[ly, lx1:lx2, 3] = 0
            break  # gestione inefficiente ma sicura; usiamo cv2 sotto

    # Versione vettoriale efficiente
    out = layer_data.copy()
    ys, xs = np.where(mask)
    ly = ys - layer_offset_y
    lx = xs - layer_offset_x
    valid = (ly >= 0) & (ly < h) & (lx >= 0) & (lx < w)
    out[ly[valid], lx[valid], 3] = 0
    return out


def fill_region(layer_data: np.ndarray,
                mask: np.ndarray,
                color: tuple[int, int, int, int],
                layer_offset_x: int = 0,
                layer_offset_y: int = 0,
                feather_radius: int = 0) -> np.ndarray:
    """Riempie la regione selezionata con un colore solido (con alpha compositing)."""
    out = layer_data.copy().astype(np.float32)
    h, w = out.shape[:2]

    if feather_radius > 0:
        alpha_mask = feather_mask(mask, feather_radius)
    else:
        alpha_mask = mask.astype(np.float32)

    r, g, b, a = color
    src_a_base = a / 255.0

    ys, xs = np.where(mask if feather_radius == 0 else (alpha_mask > 0))
    ly = ys - layer_offset_y
    lx = xs - layer_offset_x
    valid = (ly >= 0) & (ly < h) & (lx >= 0) & (lx < w)
    ly = ly[valid];  lx = lx[valid]
    ys = ys[valid];  xs = xs[valid]

    if len(ly) == 0:
        return layer_data

    src_a = alpha_mask[ys, xs] * src_a_base
    dst_a = out[ly, lx, 3] / 255.0
    out_a = src_a + dst_a * (1 - src_a)
    denom = np.maximum(out_a, 1e-8)

    for ci, cv in enumerate((r, g, b)):
        out[ly, lx, ci] = (cv * src_a + out[ly, lx, ci] * dst_a * (1 - src_a)) / denom
    out[ly, lx, 3] = np.clip(out_a * 255, 0, 255)

    return np.clip(out, 0, 255).astype(np.uint8)


# ── Marching ants — outline della selezione ───────────────────────────────────

def selection_outline_points(mask: np.ndarray,
                              max_points: int = 2000) -> list[tuple[int, int, int, int]]:
    """
    Ritorna una lista di segmenti (x1,y1,x2,y2) che formano il bordo della selezione.
    Usati per disegnare i marching ants sul canvas.
    Limitati a max_points per performance.
    """
    if not mask.any():
        return []

    # Usa cv2 findContours per i bordi della maschera bool
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    segments: list[tuple[int, int, int, int]] = []
    for contour in contours:
        pts = contour.squeeze()
        if pts.ndim < 2 or len(pts) < 2:
            continue
        n = len(pts)
        for i in range(n):
            p1 = pts[i]
            p2 = pts[(i + 1) % n]
            segments.append((int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1])))
            if len(segments) >= max_points:
                return segments
    return segments
