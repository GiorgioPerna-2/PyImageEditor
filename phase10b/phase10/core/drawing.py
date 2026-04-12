"""
core/drawing.py
---------------
Logica di disegno pixel puro su array numpy RGBA uint8.
"""
from __future__ import annotations

import math
import numpy as np


def paint_stamp(
    data:     np.ndarray,
    cx:       float,
    cy:       float,
    color:    tuple[int, int, int, int],
    size:     int,
    hardness: int,
    opacity:  int,
    mode:     str,           # "brush" | "eraser"
) -> None:
    """Applica una singola stampa circolare centrata in (cx, cy). Modifica data in-place."""
    r = max(1.0, size / 2.0)
    ix0 = max(0,             int(cx - r) - 1)
    iy0 = max(0,             int(cy - r) - 1)
    ix1 = min(data.shape[1], int(cx + r) + 2)
    iy1 = min(data.shape[0], int(cy + r) + 2)
    if ix0 >= ix1 or iy0 >= iy1:
        return

    ys = np.arange(iy0, iy1, dtype=np.float32) - cy
    xs = np.arange(ix0, ix1, dtype=np.float32) - cx
    xx, yy = np.meshgrid(xs, ys)
    dist = np.sqrt(xx * xx + yy * yy)

    hard = hardness / 100.0
    soft_start = hard * r
    if soft_start >= r - 1e-4:
        stamp_alpha = (dist <= r).astype(np.float32)
    else:
        stamp_alpha = np.where(
            dist <= soft_start, 1.0,
            np.where(dist <= r,
                     1.0 - (dist - soft_start) / max(r - soft_start, 1e-4),
                     0.0))

    stamp_alpha *= (opacity / 100.0)
    region = data[iy0:iy1, ix0:ix1]

    if mode == "eraser":
        new_alpha = np.clip(
            region[:, :, 3].astype(np.float32) - stamp_alpha * 255.0, 0, 255)
        region[:, :, 3] = new_alpha.astype(np.uint8)
    else:
        sr, sg, sb, sa_src = color
        src_a = stamp_alpha * (sa_src / 255.0)
        dst_a = region[:, :, 3].astype(np.float32) / 255.0
        out_a = src_a + dst_a * (1.0 - src_a)
        denom = np.maximum(out_a, 1e-8)
        for c_idx, c_val in enumerate((sr, sg, sb)):
            dst_c = region[:, :, c_idx].astype(np.float32)
            out_c = (c_val * src_a + dst_c * dst_a * (1.0 - src_a)) / denom
            region[:, :, c_idx] = np.clip(out_c, 0, 255).astype(np.uint8)
        region[:, :, 3] = np.clip(out_a * 255.0, 0, 255).astype(np.uint8)


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
