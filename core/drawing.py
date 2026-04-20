"""
core/drawing.py
---------------
Logica di disegno pixel puro su array numpy RGBA uint8.
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Tuple

import numpy as np

# bbox in layer-local coords: (ix0, iy0, ix1, iy1)
DirtyRect = Tuple[int, int, int, int]


# ── Maschera pennello cacheata ────────────────────────────────────────────────
# La maschera dipende solo da (size, hardness) → cache LRU così ogni
# combinazione viene calcolata una volta sola invece che ad ogni dab.
@lru_cache(maxsize=64)
def _cached_brush_mask(size: int, hardness: int) -> np.ndarray:
    """float32 (2r+1, 2r+1) alpha mask, cached per (size, hardness)."""
    r = max(1.0, size / 2.0)
    diam = int(r) * 2 + 2          # stessa formula di paint_stamp
    # coordinata relativa al centro
    half = diam / 2.0
    ys = np.arange(diam, dtype=np.float32) - (half - 0.5)
    xs = np.arange(diam, dtype=np.float32) - (half - 0.5)
    xx, yy = np.meshgrid(xs, ys)
    dist = np.sqrt(xx * xx + yy * yy)
    hard = hardness / 100.0
    soft_start = hard * r
    if soft_start >= r - 1e-4:
        mask = (dist <= r).astype(np.float32)
    else:
        mask = np.where(
            dist <= soft_start, 1.0,
            np.where(dist <= r,
                     1.0 - (dist - soft_start) / max(r - soft_start, 1e-4),
                     0.0)).astype(np.float32)
    return mask


def paint_stamp(
    data:     np.ndarray,
    cx:       float,
    cy:       float,
    color:    tuple[int, int, int, int],
    size:     int,
    hardness: int,
    opacity:  int,
    mode:     str,           # "brush" | "eraser"
) -> DirtyRect:
    """
    Applica una singola stampa circolare centrata in (cx, cy).
    Modifica data in-place e restituisce il DirtyRect (ix0,iy0,ix1,iy1)
    in coordinate layer, già clippato ai bordi dell'immagine.
    """
    r = max(1.0, size / 2.0)

    # La maschera cacheata ha dimensione fissa: diam = int(r)*2 + 2
    full_mask  = _cached_brush_mask(size, hardness)
    mask_h, mask_w = full_mask.shape

    # Centro della maschera → top-left nel layer
    cx_i = int(round(cx))
    cy_i = int(round(cy))
    half_w = mask_w // 2
    half_h = mask_h // 2

    raw_x0 = cx_i - half_w
    raw_y0 = cy_i - half_h
    raw_x1 = raw_x0 + mask_w
    raw_y1 = raw_y0 + mask_h

    ix0 = max(0, raw_x0);  iy0 = max(0, raw_y0)
    ix1 = min(data.shape[1], raw_x1);  iy1 = min(data.shape[0], raw_y1)
    if ix0 >= ix1 or iy0 >= iy1:
        return (0, 0, 0, 0)

    # Ritaglia maschera alla porzione visibile (gestisce border clipping)
    my0 = iy0 - raw_y0;  mx0 = ix0 - raw_x0
    mh  = iy1 - iy0;     mw  = ix1 - ix0
    stamp_alpha = full_mask[my0:my0 + mh, mx0:mx0 + mw] * (opacity / 100.0)

    region = data[iy0:iy1, ix0:ix1]
    assert stamp_alpha.shape == region.shape[:2], \
        f"shape mismatch: stamp={stamp_alpha.shape} region={region.shape[:2]}"

    if mode == "eraser":
        new_alpha = np.clip(
            region[:, :, 3].astype(np.float32) - stamp_alpha * 255.0, 0, 255)
        region[:, :, 3] = new_alpha.astype(np.uint8)
    else:
        sr, sg, sb, sa_src = color
        src_a  = stamp_alpha * (sa_src / 255.0)           # (h,w)
        dst_a  = region[:, :, 3].astype(np.float32) / 255.0
        out_a  = src_a + dst_a * (1.0 - src_a)
        denom  = np.maximum(out_a, 1e-8)

        # Colore sorgente come (1,1,3) per broadcast vettorizzato
        src_rgb = np.array([sr, sg, sb], dtype=np.float32).reshape(1, 1, 3)
        dst_rgb = region[:, :, :3].astype(np.float32)

        # Porter-Duff over su tutti e tre i canali in un colpo solo
        out_rgb = (src_rgb * src_a[:, :, np.newaxis]
                   + dst_rgb * (dst_a * (1.0 - src_a))[:, :, np.newaxis]) / denom[:, :, np.newaxis]

        region[:, :, :3] = np.clip(out_rgb, 0, 255).astype(np.uint8)
        region[:, :, 3]  = np.clip(out_a * 255.0, 0, 255).astype(np.uint8)

    return (ix0, iy0, ix1, iy1)


def interpolate_stroke(
    p0: tuple[float, float],
    p1: tuple[float, float],
    size: int,
    spacing_factor: float = 0.25,
) -> list[tuple[float, float]]:
    """Punti di stamp tra p0 e p1, distanziati di spacing_factor * size."""
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    dist = math.hypot(dx, dy)
    step = max(1.0, size * spacing_factor)
    if dist < step:
        return [p1]
    n = int(dist / step)
    return [(p0[0] + dx * i / n, p0[1] + dy * i / n) for i in range(1, n + 1)]
