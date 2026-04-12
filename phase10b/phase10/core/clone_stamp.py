"""
core/clone_stamp.py
-------------------
Timbro clone: copia pixel da un'area sorgente all'area di destinazione.
L'offset sorgente/destinazione viene mantenuto dal chiamante.
"""
from __future__ import annotations

import math
import numpy as np
#from .drawing import _clip   # riusa il helper di drawing.py


def clone_stamp_dab(
    src_data: np.ndarray,     # layer sorgente (può essere lo stesso di dst)
    dst_data: np.ndarray,     # layer destinazione (modificato in-place)
    src_x: float, src_y: float,
    dst_x: float, dst_y: float,
    size: int,
    hardness: float = 0.8,    # 0.0 = morbido, 1.0 = bordo netto
    opacity: float = 1.0,     # 0.0–1.0
) -> None:
    """
    Applica un singolo dab del timbro clone:
    copia la patch centrata in (src_x, src_y) del sorgente
    e la incolla centrata in (dst_x, dst_y) della destinazione
    usando normal alpha compositing con softness e opacity.
    """
    r = max(1, size // 2)
    sh, sw = src_data.shape[:2]
    dh, dw = dst_data.shape[:2]

    # Genera maschera circolare (float32, dimensione 2r+1)
    diam = 2 * r + 1
    yy, xx = np.mgrid[:diam, :diam]
    dist = np.sqrt(((xx - r) ** 2 + (yy - r) ** 2).astype(np.float32))
    inner = r * max(0.0, min(1.0, hardness))
    mask = np.zeros(dist.shape, dtype=np.float32)
    mask[dist <= inner] = 1.0
    soft = (dist > inner) & (dist <= r)
    if soft.any() and r > inner:
        mask[soft] = 1.0 - (dist[soft] - inner) / max(r - inner, 1e-6)
    mask *= opacity

    # Intersezione del dab con sorgente
    sx0 = int(round(src_x)) - r;  sy0 = int(round(src_y)) - r
    sx1 = sx0 + diam;              sy1 = sy0 + diam

    csx0 = max(0, sx0);  csy0 = max(0, sy0)
    csx1 = min(sw, sx1); csy1 = min(sh, sy1)
    if csx0 >= csx1 or csy0 >= csy1:
        return

    mx0 = csx0 - sx0;  my0 = csy0 - sy0
    mx1 = mx0 + (csx1 - csx0); my1 = my0 + (csy1 - csy0)

    # Intersezione del dab con destinazione
    dx0 = int(round(dst_x)) - r;  dy0 = int(round(dst_y)) - r
    dx1 = dx0 + diam;              dy1 = dy0 + diam

    cdx0 = max(0, dx0);  cdy0 = max(0, dy0)
    cdx1 = min(dw, dx1); cdy1 = min(dh, dy1)
    if cdx0 >= cdx1 or cdy0 >= cdy1:
        return

    mdx0 = cdx0 - dx0;  mdy0 = cdy0 - dy0
    mdx1 = mdx0 + (cdx1 - cdx0); mdy1 = mdy0 + (cdy1 - cdy0)

    # Clip al minimo comune (src e dst possono avere dimensioni diverse)
    rh = min(my1 - my0, mdy1 - mdy0)
    rw = min(mx1 - mx0, mdx1 - mdx0)
    if rh <= 0 or rw <= 0:
        return

    src_patch = src_data[csy0:csy0 + rh, csx0:csx0 + rw].astype(np.float32)
    dst_patch = dst_data[cdy0:cdy0 + rh, cdx0:cdx0 + rw].astype(np.float32)
    m = mask[my0:my0 + rh, mx0:mx0 + rw]

    src_a = src_patch[:, :, 3:4] / 255.0 * m[:, :, np.newaxis]
    dst_a = dst_patch[:, :, 3:4] / 255.0
    out_a = src_a + dst_a * (1 - src_a)
    denom = np.maximum(out_a, 1e-8)

    out = dst_patch.copy()
    out[:, :, :3] = (src_patch[:, :, :3] * src_a +
                     dst_patch[:, :, :3] * dst_a * (1 - src_a)) / denom
    out[:, :, 3:4] = out_a * 255.0

    dst_data[cdy0:cdy0 + rh, cdx0:cdx0 + rw] = np.clip(out, 0, 255).astype(np.uint8)


def interpolate_clone_stroke(
    p0: tuple[float, float], p1: tuple[float, float], size: int
) -> list[tuple[float, float]]:
    """Punti di stamp tra p0 e p1, distanziati di 0.25 * size."""
    dx = p1[0] - p0[0];  dy = p1[1] - p0[1]
    dist = math.hypot(dx, dy)
    step = max(1.0, size * 0.25)
    if dist < step:
        return [p1]
    n = int(dist / step)
    return [(p0[0] + dx * i / n, p0[1] + dy * i / n) for i in range(1, n + 1)]
