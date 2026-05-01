"""
core/drawing.py
---------------
Logica di disegno pixel puro su array numpy RGBA uint8.

Tipi di pennello supportati
---------------------------
  brush       — pennello morbido standard
  eraser      — gomma
  airbrush    — aerografo: particelle sparse con scatter casuale
  pencil      — matita: texture granosa, hardness alta di default
  calligraphy — punta piatta/ellittica ruotata (stile calligrafia / pennino)

Punte (tip) per brush / pencil / calligraphy
---------------------------------------------
  round       — cerchio (default)
  calligraphy — ellisse ruotata (angolo impostabile)
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Tuple

import numpy as np

# bbox in layer-local coords: (ix0, iy0, ix1, iy1)
DirtyRect = Tuple[int, int, int, int]

# ── Maschere cacheate ─────────────────────────────────────────────────────────

@lru_cache(maxsize=128)
def _cached_brush_mask(size: int, hardness: int) -> np.ndarray:
    """Maschera circolare float32, cached per (size, hardness)."""
    r = max(1.0, size / 2.0)
    diam = int(r) * 2 + 2
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


@lru_cache(maxsize=128)
def _cached_calligraphy_mask(size: int, hardness: int,
                              angle_deg: int, ratio: int) -> np.ndarray:
    """
    Maschera ellittica ruotata per la punta calligrafica.
    angle_deg  — angolo della punta in gradi (0 = orizzontale)
    ratio      — rapporto asse_corto/asse_lungo × 100 (es. 25 → 0.25)
    """
    r = max(1.0, size / 2.0)
    diam = int(r) * 2 + 2
    half = diam / 2.0
    ys = np.arange(diam, dtype=np.float32) - (half - 0.5)
    xs = np.arange(diam, dtype=np.float32) - (half - 0.5)
    xx, yy = np.meshgrid(xs, ys)

    # Ruota le coordinate
    rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    rx_rot =  xx * cos_a + yy * sin_a    # asse lungo
    ry_rot = -xx * sin_a + yy * cos_a    # asse corto

    rx = r
    ry = max(1.0, r * (ratio / 100.0))
    dist = np.sqrt((rx_rot / rx) ** 2 + (ry_rot / ry) ** 2)

    hard = hardness / 100.0
    soft_start = hard
    if soft_start >= 1.0 - 1e-4:
        mask = (dist <= 1.0).astype(np.float32)
    else:
        mask = np.where(
            dist <= soft_start, 1.0,
            np.where(dist <= 1.0,
                     1.0 - (dist - soft_start) / max(1.0 - soft_start, 1e-4),
                     0.0)).astype(np.float32)
    return mask


@lru_cache(maxsize=128)
def _cached_pencil_noise(size: int, seed: int = 42) -> np.ndarray:
    """
    Texture granosa per la matita — aspetto carta/grafite.
    Usa una distribuzione bimodale: la maggior parte dei pixel è zero o quasi,
    pochi hanno alta densità → tratto visibilmente discontinuo rispetto al pennello.
    """
    r = max(1.0, size / 2.0)
    diam = int(r) * 2 + 2
    rng = np.random.default_rng(seed)
    # Distribuzione power-law: la maggior parte → vicino a zero
    noise = rng.power(0.4, (diam, diam)).astype(np.float32)
    # Aggiungi variazione ad alta frequenza (texture carta)
    grain = rng.uniform(0.0, 0.35, (diam, diam)).astype(np.float32)
    noise = np.clip(noise * 0.85 + grain, 0.0, 1.0)
    # Maschera circolare come inviluppo (hardness=85 per bordo morbido)
    base = _cached_brush_mask(size, 85)
    return (noise * base).astype(np.float32)


# ── Funzione principale di stampa ─────────────────────────────────────────────

def paint_stamp(
    data:       np.ndarray,
    cx:         float,
    cy:         float,
    color:      tuple[int, int, int, int],
    size:       int,
    hardness:   int,
    opacity:    int,
    mode:       str,           # "brush"|"eraser"|"airbrush"|"pencil"|"calligraphy"
    tip:        str = "round", # "round"|"calligraphy"
    tip_angle:  int = 45,      # gradi (0–180), usato se tip=="calligraphy"
    tip_ratio:  int = 25,      # rapporto asse corto/lungo ×100 (5–95)
) -> DirtyRect:
    """
    Applica un dab centrato in (cx, cy).
    Restituisce DirtyRect (ix0,iy0,ix1,iy1) in coordinate layer.
    """
    if mode == "airbrush":
        return _paint_airbrush(data, cx, cy, color, size, opacity)

    # Scegli maschera base
    if tip == "calligraphy" or mode == "calligraphy":
        full_mask = _cached_calligraphy_mask(
            size, hardness,
            tip_angle % 180,
            max(5, min(95, tip_ratio)))
    else:
        full_mask = _cached_brush_mask(size, hardness)

    if mode == "pencil":
        # Sovrappone la texture granosa sulla maschera
        noise = _cached_pencil_noise(size)
        # Usa il minore dei due per mantenere la forma
        full_mask = np.minimum(full_mask, noise * 1.8)

    mask_h, mask_w = full_mask.shape
    cx_i = int(round(cx)); cy_i = int(round(cy))
    half_w = mask_w // 2;  half_h = mask_h // 2

    raw_x0 = cx_i - half_w;  raw_y0 = cy_i - half_h
    raw_x1 = raw_x0 + mask_w; raw_y1 = raw_y0 + mask_h

    ix0 = max(0, raw_x0);  iy0 = max(0, raw_y0)
    ix1 = min(data.shape[1], raw_x1);  iy1 = min(data.shape[0], raw_y1)
    if ix0 >= ix1 or iy0 >= iy1:
        return (0, 0, 0, 0)

    my0 = iy0 - raw_y0;  mx0 = ix0 - raw_x0
    mh  = iy1 - iy0;     mw  = ix1 - ix0
    stamp_alpha = full_mask[my0:my0 + mh, mx0:mx0 + mw] * (opacity / 100.0)

    region = data[iy0:iy1, ix0:ix1]

    if mode == "eraser":
        new_alpha = np.clip(
            region[:, :, 3].astype(np.float32) - stamp_alpha * 255.0, 0, 255)
        region[:, :, 3] = new_alpha.astype(np.uint8)
    else:
        sr, sg, sb, sa_src = color
        src_a  = stamp_alpha * (sa_src / 255.0)
        dst_a  = region[:, :, 3].astype(np.float32) / 255.0
        out_a  = src_a + dst_a * (1.0 - src_a)
        denom  = np.maximum(out_a, 1e-8)
        src_rgb = np.array([sr, sg, sb], dtype=np.float32).reshape(1, 1, 3)
        dst_rgb = region[:, :, :3].astype(np.float32)
        out_rgb = (src_rgb * src_a[:, :, np.newaxis]
                   + dst_rgb * (dst_a * (1.0 - src_a))[:, :, np.newaxis]) / denom[:, :, np.newaxis]
        region[:, :, :3] = np.clip(out_rgb, 0, 255).astype(np.uint8)
        region[:, :, 3]  = np.clip(out_a * 255.0, 0, 255).astype(np.uint8)

    return (ix0, iy0, ix1, iy1)


def _paint_airbrush(
    data:    np.ndarray,
    cx:      float,
    cy:      float,
    color:   tuple[int, int, int, int],
    size:    int,
    opacity: int,
) -> DirtyRect:
    """
    Aerografo: N particelle sparse nel raggio, con caduta off-center.
    Ogni particella è un singolo pixel con opacità ridotta — effetto spray.
    """
    h, w = data.shape[:2]
    r = max(2.0, size / 2.0)

    # Numero di particelle proporzionale all'area (con un cap sensato)
    n_particles = max(4, min(int(math.pi * r * r * 0.18 * opacity / 100.0), 600))

    # Genera coordinate polari casuali con distribuzione gaussiana
    rng = np.random.default_rng()   # non cacheato: ogni dab deve essere diverso
    angles = rng.uniform(0, 2 * math.pi, n_particles)
    # Distanza: distribuzione quadratica per densità centrale
    radii  = r * np.sqrt(rng.uniform(0, 1, n_particles))

    xs = (cx + radii * np.cos(angles)).astype(int)
    ys = (cy + radii * np.sin(angles)).astype(int)

    # Filtra fuori canvas
    mask = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    xs, ys = xs[mask], ys[mask]
    radii  = radii[mask]

    if len(xs) == 0:
        return (0, 0, 0, 0)

    # Opacità per particella: più bassa verso il bordo
    dist_norm = radii / r
    p_alpha = ((1.0 - dist_norm ** 1.5) * opacity / 100.0 * 0.6).clip(0, 1)

    sr, sg, sb, sa_src = color
    base_a = sa_src / 255.0

    for i in range(len(xs)):
        x, y = int(xs[i]), int(ys[i])
        src_a_val = float(p_alpha[i]) * base_a
        if src_a_val < 0.002:
            continue
        dst = data[y, x].astype(np.float32)
        dst_a_val = dst[3] / 255.0
        out_a_val = src_a_val + dst_a_val * (1.0 - src_a_val)
        if out_a_val < 1e-8:
            continue
        data[y, x, 0] = int(np.clip((sr * src_a_val + dst[0] * dst_a_val * (1.0 - src_a_val)) / out_a_val, 0, 255))
        data[y, x, 1] = int(np.clip((sg * src_a_val + dst[1] * dst_a_val * (1.0 - src_a_val)) / out_a_val, 0, 255))
        data[y, x, 2] = int(np.clip((sb * src_a_val + dst[2] * dst_a_val * (1.0 - src_a_val)) / out_a_val, 0, 255))
        data[y, x, 3] = int(np.clip(out_a_val * 255.0, 0, 255))

    x0 = max(0, int(cx - r)); y0 = max(0, int(cy - r))
    x1 = min(w, int(cx + r) + 1); y1 = min(h, int(cy + r) + 1)
    return (x0, y0, x1, y1)


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
