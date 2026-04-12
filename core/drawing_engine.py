"""
drawing_engine.py — motore pixel per gli strumenti di disegno.

Tutti i metodi lavorano su array RGBA uint8 (H, W, 4) e modificano
il layer in-place restituendo il bounding-box delle pixel toccate,
così il caller può limitare gli aggiornamenti.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Tipo helper
RGBA = Tuple[int, int, int, int]
BBox = Tuple[int, int, int, int]   # x, y, w, h

# ── Brush mask ────────────────────────────────────────────────────────────────

def _make_brush(size: int, hardness: float) -> np.ndarray:
    """
    Genera una maschera di pennello circolare float32 [0,1] di dimensione (size, size).
    hardness 0.0 = sfumato al massimo, 1.0 = bordo netto.
    """
    size = max(1, int(size))
    if size == 1:
        return np.ones((1, 1), dtype=np.float32)

    r = size / 2.0
    y, x = np.ogrid[:size, :size]
    dist = np.sqrt((x - r + 0.5) ** 2 + (y - r + 0.5) ** 2) / r

    # Hardness: transizione morbida tra (hardness) e 1.0
    h = np.clip(hardness, 0.0, 1.0)
    inner = h * 0.95          # raggio interno dove alpha = 1
    mask = np.clip(1.0 - (dist - inner) / max(1.0 - inner, 1e-6), 0.0, 1.0)
    mask = mask ** 0.5        # gamma per bordi più naturali
    return mask.astype(np.float32)


# ── Funzione di stampa del pennello ──────────────────────────────────────────

def _stamp(
    data: np.ndarray,
    cx: float, cy: float,
    mask: np.ndarray,
    color_rgba: RGBA,
    opacity: float,
    erase: bool = False,
) -> Optional[BBox]:
    """
    Applica una singola 'stampa' del pennello centrata in (cx, cy).
    Se erase=True azzera l'alpha invece di dipingere.
    Restituisce (x,y,w,h) delle celle modificate o None se fuori canvas.
    """
    h, w = data.shape[:2]
    ms = mask.shape[0]
    half = ms / 2.0

    # Bounding box della stampa nel canvas
    x0 = int(math.floor(cx - half))
    y0 = int(math.floor(cy - half))
    x1 = x0 + ms
    y1 = y0 + ms

    # Intersezione con canvas
    cx0 = max(0, x0);  cy0 = max(0, y0)
    cx1 = min(w, x1);  cy1 = min(h, y1)
    if cx0 >= cx1 or cy0 >= cy1:
        return None

    # Regione della maschera corrispondente
    mx0 = cx0 - x0;  my0 = cy0 - y0
    mx1 = mx0 + (cx1 - cx0);  my1 = my0 + (cy1 - cy0)

    sub_mask = mask[my0:my1, mx0:mx1]          # (rh, rw)
    alpha_factor = sub_mask * opacity           # [0,1]

    region = data[cy0:cy1, cx0:cx1].astype(np.float32)

    if erase:
        # Riduce alpha
        region[:, :, 3] *= (1.0 - alpha_factor)
    else:
        r, g, b, a = color_rgba
        src_a = alpha_factor * (a / 255.0)
        dst_a = region[:, :, 3] / 255.0
        out_a = src_a + dst_a * (1.0 - src_a)
        denom = np.maximum(out_a, 1e-8)
        for c_idx, c_val in enumerate((r, g, b)):
            region[:, :, c_idx] = (
                c_val * src_a + region[:, :, c_idx] * dst_a * (1.0 - src_a)
            ) / denom
        region[:, :, 3] = out_a * 255.0

    data[cy0:cy1, cx0:cx1] = np.clip(region, 0, 255).astype(np.uint8)
    return (cx0, cy0, cx1 - cx0, cy1 - cy0)


# ── Tratto interpolato ────────────────────────────────────────────────────────

def stroke_line(
    data: np.ndarray,
    x0: float, y0: float,
    x1: float, y1: float,
    mask: np.ndarray,
    color_rgba: RGBA,
    opacity: float,
    erase: bool = False,
) -> Optional[BBox]:
    """Interpola tra due punti e applica stamp ogni mezzo pixel."""
    dist = math.hypot(x1 - x0, y1 - y0)
    steps = max(1, int(dist * 2))
    boxes = []
    for i in range(steps + 1):
        t = i / steps
        px = x0 + t * (x1 - x0)
        py = y0 + t * (y1 - y0)
        b = _stamp(data, px, py, mask, color_rgba, opacity, erase)
        if b:
            boxes.append(b)
    if not boxes:
        return None
    xs = [b[0] for b in boxes];  ys = [b[1] for b in boxes]
    x2s = [b[0]+b[2] for b in boxes];  y2s = [b[1]+b[3] for b in boxes]
    return (min(xs), min(ys), max(x2s)-min(xs), max(y2s)-min(ys))


# ── Sfumino ───────────────────────────────────────────────────────────────────

def smudge_line(
    data: np.ndarray,
    x0: float, y0: float,
    x1: float, y1: float,
    size: int,
    strength: float = 0.7,
) -> Optional[BBox]:
    """
    Effetto sfumino: campiona i pixel sotto il pennello e li 'trascina'
    nella direzione del movimento mescolando con i pixel adiacenti.
    """
    h, w = data.shape[:2]
    r = max(1, size // 2)
    dist = math.hypot(x1 - x0, y1 - y0)
    steps = max(1, int(dist * 2))
    boxes = []

    for i in range(1, steps + 1):
        t = i / steps
        cx = int(x0 + t * (x1 - x0))
        cy = int(y0 + t * (y1 - y0))

        # Patch sorgente (un po' più indietro)
        bt = max(0.0, (i - 1) / steps)
        sx = int(x0 + bt * (x1 - x0))
        sy = int(y0 + bt * (y1 - y0))

        for patch_cx, patch_cy, is_dst in ((cx, cy, True), (sx, sy, False)):
            px0 = max(0, patch_cx - r);  py0 = max(0, patch_cy - r)
            px1 = min(w, patch_cx + r);  py1 = min(h, patch_cy + r)
            if px0 >= px1 or py0 >= py1:
                continue
            boxes.append((px0, py0, px1-px0, py1-py0))

        # Regione destinazione
        dx0 = max(0, cx - r);  dy0 = max(0, cy - r)
        dx1 = min(w, cx + r);  dy1 = min(h, cy + r)
        sx0 = max(0, sx - r);  sy0 = max(0, sy - r)
        sx1 = min(w, sx + r);  sy1 = min(h, sy + r)

        dw = min(dx1-dx0, sx1-sx0);  dh = min(dy1-dy0, sy1-sy0)
        if dw <= 0 or dh <= 0:
            continue

        src_patch = data[sy0:sy0+dh, sx0:sx0+dw].astype(np.float32)
        dst_patch = data[dy0:dy0+dh, dx0:dx0+dw].astype(np.float32)
        blended = src_patch * strength + dst_patch * (1.0 - strength)
        data[dy0:dy0+dh, dx0:dx0+dw] = np.clip(blended, 0, 255).astype(np.uint8)

    if not boxes:
        return None
    xs = [b[0] for b in boxes];  ys = [b[1] for b in boxes]
    x2s = [b[0]+b[2] for b in boxes];  y2s = [b[1]+b[3] for b in boxes]
    return (min(xs), min(ys), max(x2s)-min(xs), max(y2s)-min(ys))


# ── Flood fill ────────────────────────────────────────────────────────────────

def flood_fill(
    data: np.ndarray,
    x: int, y: int,
    color_rgba: RGBA,
    tolerance: int = 32,
) -> BBox:
    """
    Flood fill BFS sul layer a partire da (x,y) con la tolleranza data.
    Riempie nell'alpha composito (rispetta il canale alpha esistente).
    """
    h, w = data.shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        return (0, 0, 0, 0)

    target = data[y, x].copy().astype(np.int32)
    fill_color = np.array(color_rgba, dtype=np.uint8)

    # Se già uguale, non fare niente
    if np.all(np.abs(target - fill_color.astype(np.int32)) <= 1):
        return (0, 0, 0, 0)

    visited = np.zeros((h, w), dtype=bool)
    queue = deque()
    queue.append((x, y))
    visited[y, x] = True

    min_x, min_y, max_x, max_y = x, y, x, y

    while queue:
        cx, cy = queue.popleft()
        pixel = data[cy, cx].astype(np.int32)
        if np.max(np.abs(pixel - target)) > tolerance:
            continue

        data[cy, cx] = fill_color
        if cx < min_x: min_x = cx
        if cy < min_y: min_y = cy
        if cx > max_x: max_x = cx
        if cy > max_y: max_y = cy

        for nx, ny in ((cx+1,cy),(cx-1,cy),(cx,cy+1),(cx,cy-1)):
            if 0 <= nx < w and 0 <= ny < h and not visited[ny, nx]:
                visited[ny, nx] = True
                queue.append((nx, ny))

    return (min_x, min_y, max_x - min_x + 1, max_y - min_y + 1)


# ── Contagocce ────────────────────────────────────────────────────────────────

def pick_color(data: np.ndarray, x: int, y: int) -> Optional[RGBA]:
    """Restituisce il colore RGBA del pixel in (x,y) o None se fuori canvas."""
    h, w = data.shape[:2]
    if 0 <= x < w and 0 <= y < h:
        r, g, b, a = data[y, x]
        return int(r), int(g), int(b), int(a)
    return None


# ── Testo ─────────────────────────────────────────────────────────────────────

def draw_text(
    data: np.ndarray,
    x: int, y: int,
    text: str,
    color_rgba: RGBA,
    font_family: str = "Arial",
    font_size: int = 24,
) -> BBox:
    """
    Rende 'text' sul layer a partire da (x,y) usando Pillow.
    Restituisce il bounding box del testo disegnato.
    """
    h, w = data.shape[:2]
    if not text:
        return (0, 0, 0, 0)

    # Prova a caricare il font, fallback su default
    try:
        font = ImageFont.truetype(font_family + ".ttf", font_size)
    except Exception:
        try:
            import os
            # Cerca il font nelle cartelle di sistema
            candidates = [
                f"/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                f"/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                f"/System/Library/Fonts/Helvetica.ttc",
                f"C:/Windows/Fonts/arial.ttf",
            ]
            font = None
            for c in candidates:
                if os.path.exists(c):
                    font = ImageFont.truetype(c, font_size)
                    break
            if font is None:
                font = ImageFont.load_default()
        except Exception:
            font = ImageFont.load_default()

    # Misura il testo
    tmp_img = Image.new("RGBA", (1, 1))
    tmp_draw = ImageDraw.Draw(tmp_img)
    bbox = tmp_draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    # Area di rendering (clippata al canvas)
    rx = max(0, x);  ry = max(0, y)
    rx2 = min(w, x + tw + 4);  ry2 = min(h, y + th + 4)
    if rx >= rx2 or ry >= ry2:
        return (0, 0, 0, 0)

    # Disegna su patch temporanea
    pw = rx2 - rx;  ph = ry2 - ry
    patch_img = Image.new("RGBA", (pw, ph), (0, 0, 0, 0))
    draw = ImageDraw.Draw(patch_img)
    draw.text((0 - (rx - x), 0 - (ry - y)), text, font=font,
              fill=color_rgba)

    patch_arr = np.array(patch_img, dtype=np.uint8)
    existing = data[ry:ry2, rx:rx2].astype(np.float32)

    src_a = patch_arr[:, :, 3:4] / 255.0
    dst_a = existing[:, :, 3:4] / 255.0
    out_a = src_a + dst_a * (1.0 - src_a)
    denom = np.maximum(out_a, 1e-8)
    for ci in range(3):
        existing[:, :, ci] = (
            patch_arr[:, :, ci].astype(np.float32) * src_a[:, :, 0]
            + existing[:, :, ci] * dst_a[:, :, 0] * (1.0 - src_a[:, :, 0])
        ) / denom[:, :, 0]
    existing[:, :, 3] = out_a[:, :, 0] * 255.0
    data[ry:ry2, rx:rx2] = np.clip(existing, 0, 255).astype(np.uint8)

    return (rx, ry, pw, ph)
