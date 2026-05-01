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

from PyQt6.QtCore import Qt, QPointF
from PyQt6.QtGui import (
    QImage, QPainter, QFont, QColor, QPen, QBrush,
    QFontMetrics, QPainterPath, QLinearGradient,
)

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

# ── Testo: render Qt-based (supporta bold, italic, multilinea, opacità, effetti) ──

def _make_qfont(family: str, size: int, bold: bool, italic: bool) -> QFont:
    f = QFont(family)
    f.setPixelSize(max(4, size))
    f.setBold(bold)
    f.setItalic(italic)
    return f


def _paint_text_on(
    painter: QPainter,
    lines: list[str],
    font: QFont,
    fm: QFontMetrics,
    ox: float, oy: float,
    color: QColor,
    opacity: float,
    effect: str,
    effect_color: QColor,
    outline_width: int,
    img_w: int,
    img_h: int,
) -> None:
    """
    Disegna il testo (già diviso in righe) su un QPainter aperto.
    ox, oy sono le coordinate top-left dell'area testo nell'immagine.
    Usata sia da render_text_patch (QImage) sia dall'anteprima canvas (viewport).
    """
    line_sp = fm.lineSpacing()
    painter.setFont(font)

    if effect == "shadow":
        sdx = max(3, font.pixelSize() // 10)
        sdy = max(3, font.pixelSize() // 10)
        shadow_c = QColor(effect_color.red(), effect_color.green(),
                          effect_color.blue(),
                          min(255, int(effect_color.alpha() * opacity * 0.65)))
        for i, line in enumerate(lines):
            painter.setPen(shadow_c)
            painter.setOpacity(1.0)
            painter.drawText(QPointF(ox + sdx, oy + fm.ascent() + i * line_sp + sdy), line)
        for i, line in enumerate(lines):
            painter.setPen(QColor(color.red(), color.green(), color.blue(),
                                  min(255, int(color.alpha() * opacity))))
            painter.setOpacity(1.0)
            painter.drawText(QPointF(ox, oy + fm.ascent() + i * line_sp), line)

    elif effect == "outline":
        fill_c = QColor(color.red(), color.green(), color.blue(),
                        min(255, int(color.alpha() * opacity)))
        stroke_c = QColor(effect_color.red(), effect_color.green(),
                          effect_color.blue(),
                          min(255, int(effect_color.alpha() * opacity)))
        for i, line in enumerate(lines):
            if not line:
                continue
            path = QPainterPath()
            path.addText(QPointF(ox, oy + fm.ascent() + i * line_sp), font, line)
            pen = QPen(stroke_c, outline_width * 2)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)
            painter.setBrush(QBrush(fill_c))
            painter.setOpacity(1.0)
            painter.drawPath(path)

    elif effect == "gradient":
        c2 = QColor(effect_color.red(), effect_color.green(), effect_color.blue(),
                    min(255, int(effect_color.alpha() * opacity)))
        c1 = QColor(color.red(), color.green(), color.blue(),
                    min(255, int(color.alpha() * opacity)))
        for i, line in enumerate(lines):
            if not line:
                continue
            lw = fm.horizontalAdvance(line) or 1
            grad = QLinearGradient(QPointF(ox, 0.0), QPointF(ox + lw, 0.0))
            grad.setColorAt(0.0, c1)
            grad.setColorAt(1.0, c2)
            path = QPainterPath()
            path.addText(QPointF(ox, oy + fm.ascent() + i * line_sp), font, line)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(grad))
            painter.setOpacity(1.0)
            painter.drawPath(path)

    elif effect == "arch":
        flat = " ".join(lines)
        _paint_arch(painter, flat, font, fm, img_w, img_h, color, opacity)

    elif effect == "wave":
        flat = " ".join(lines)
        wave_amp = int(fm.height() * 0.38)
        _paint_wave(painter, flat, font, fm, ox, oy + wave_amp, color, opacity)

    else:  # "none"
        main_c = QColor(color.red(), color.green(), color.blue(),
                        min(255, int(color.alpha() * opacity)))
        painter.setPen(main_c)
        painter.setOpacity(1.0)
        for i, line in enumerate(lines):
            painter.drawText(QPointF(ox, oy + fm.ascent() + i * line_sp), line)


def _paint_arch(painter: QPainter, text: str, font: QFont,
                fm: QFontMetrics, img_w: int, img_h: int,
                color: QColor, opacity: float) -> None:
    if not text:
        return
    total_w = fm.horizontalAdvance(text) or 1
    arc_span = math.pi * 0.72        # ~130° di arco
    radius = total_w / arc_span
    cx = img_w / 2.0
    cy = img_h / 2.0 + radius * 0.55

    c = QColor(color.red(), color.green(), color.blue(),
               min(255, int(color.alpha() * opacity)))
    painter.setOpacity(1.0)

    x_cur = -total_w / 2.0
    for ch in text:
        cw = fm.horizontalAdvance(ch)
        angle = (x_cur + cw / 2.0) / radius
        px = cx + radius * math.sin(angle)
        py = cy - radius * math.cos(angle)
        painter.save()
        painter.translate(px, py)
        painter.rotate(math.degrees(angle))
        painter.setPen(c)
        painter.setFont(font)
        painter.drawText(QPointF(-cw / 2.0, 0.0), ch)
        painter.restore()
        x_cur += cw


def _paint_wave(painter: QPainter, text: str, font: QFont,
                fm: QFontMetrics, ox: float, oy: float,
                color: QColor, opacity: float) -> None:
    if not text:
        return
    amplitude = fm.height() * 0.38
    total_w = fm.horizontalAdvance(text) or 1
    freq = 2.0 * math.pi * 2.2 / total_w

    c = QColor(color.red(), color.green(), color.blue(),
               min(255, int(color.alpha() * opacity)))
    painter.setOpacity(1.0)
    painter.setPen(c)
    painter.setFont(font)

    cx = float(ox)
    base = oy + fm.ascent()
    for ch in text:
        cw = fm.horizontalAdvance(ch)
        dy = amplitude * math.sin(freq * (cx - ox + cw / 2.0))
        painter.drawText(QPointF(cx, base + dy), ch)
        cx += cw


def render_text_patch(
    text: str,
    font_family: str,
    font_size: int,
    bold: bool = False,
    italic: bool = False,
    color_rgba: RGBA = (0, 0, 0, 255),
    opacity: float = 1.0,
    effect: str = "none",
    effect_color: Optional[RGBA] = None,
    outline_width: int = 2,
    rotation: float = 0.0,
) -> Tuple[np.ndarray, int, int]:
    """
    Renderizza il testo in un array RGBA numpy.

    Returns
    -------
    (patch, anchor_x, anchor_y)
        Per posizionare la patch in modo che il punto ancora (x, y) corrisponda
        alla top-left del testo, disegna la patch a (x - anchor_x, y - anchor_y).
    """
    font = _make_qfont(font_family, font_size, bold, italic)
    fm = QFontMetrics(font)

    lines = text.splitlines() or [""]
    line_sp  = fm.lineSpacing()
    max_w    = max((fm.horizontalAdvance(ln) for ln in lines), default=1)
    text_h   = line_sp * len(lines)

    c1 = QColor(*color_rgba[:3], min(255, color_rgba[3]))
    ec_rgba  = effect_color if effect_color else (0, 0, 0, 200)
    c2 = QColor(*ec_rgba[:3], min(255, ec_rgba[3]))

    # Calcola padding basato sull'effetto
    sdx = sdy = 0
    if effect == "shadow":
        sdx = max(3, font_size // 10)
        sdy = max(3, font_size // 10)
        pad = max(sdx, sdy) + 4
    elif effect == "outline":
        pad = outline_width + 4
    elif effect in ("arch", "wave"):
        pad = font_size // 2 + 8
    else:
        pad = 4

    if effect == "arch":
        side = max(max_w + pad * 2, text_h + font_size * 2 + pad * 2)
        img_w, img_h = side, side
        anchor_x = img_w // 2 - max_w // 2
        anchor_y = pad
    elif effect == "wave":
        wave_amp = int(fm.height() * 0.38)
        img_w = max_w + pad * 2
        img_h = text_h + wave_amp * 2 + pad * 2
        anchor_x = pad
        anchor_y = wave_amp + pad
    else:
        img_w = max_w + pad * 2 + sdx
        img_h = text_h + pad * 2 + sdy
        anchor_x = pad
        anchor_y = pad

    img_w = max(img_w, 4)
    img_h = max(img_h, 4)

    img = QImage(img_w, img_h, QImage.Format.Format_RGBA8888)
    img.fill(Qt.GlobalColor.transparent)
    painter = QPainter(img)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)

    _paint_text_on(painter, lines, font, fm,
                   float(anchor_x), float(anchor_y),
                   c1, opacity, effect, c2, outline_width,
                   img_w, img_h)
    painter.end()

    # Applica rotazione se richiesta
    if rotation != 0.0:
        from PyQt6.QtGui import QTransform
        t = QTransform()
        t.translate(anchor_x, anchor_y)
        t.rotate(rotation)
        t.translate(-anchor_x, -anchor_y)
        img = img.transformed(t, Qt.TransformationMode.SmoothTransformation)
        iw2, ih2 = img.width(), img.height()
        anchor_x = iw2 // 2 - (img_w // 2 - anchor_x)
        anchor_y = ih2 // 2 - (img_h // 2 - anchor_y)

    img_r = img.convertToFormat(QImage.Format.Format_RGBA8888)
    ptr   = img_r.bits()
    ptr.setsize(img_r.sizeInBytes())
    arr   = np.frombuffer(ptr, dtype=np.uint8).reshape(
        (img_r.height(), img_r.width(), 4)).copy()
    return arr, anchor_x, anchor_y


def _alpha_composite(dst: np.ndarray, src: np.ndarray, px: int, py: int) -> None:
    dh, dw = dst.shape[:2]
    sh, sw = src.shape[:2]
    x0, y0 = max(0, px), max(0, py)
    x1, y1 = min(dw, px + sw), min(dh, py + sh)
    if x0 >= x1 or y0 >= y1:
        return
    s = src[y0 - py:y1 - py, x0 - px:x1 - px].astype(np.float32)
    d = dst[y0:y1, x0:x1].astype(np.float32)
    sa = s[:, :, 3:4] / 255.0
    da = d[:, :, 3:4] / 255.0
    oa = sa + da * (1.0 - sa)
    denom = np.maximum(oa, 1e-8)
    for ci in range(3):
        d[:, :, ci] = (s[:, :, ci] * sa[:, :, 0]
                       + d[:, :, ci] * da[:, :, 0] * (1.0 - sa[:, :, 0])) / denom[:, :, 0]
    d[:, :, 3] = oa[:, :, 0] * 255.0
    dst[y0:y1, x0:x1] = np.clip(d, 0, 255).astype(np.uint8)


def draw_text(
    data: np.ndarray,
    x: int, y: int,
    text: str,
    color_rgba: RGBA,
    font_family: str  = "Arial",
    font_size:   int  = 24,
    bold:        bool = False,
    italic:      bool = False,
    opacity:     float = 1.0,
    effect:      str   = "none",
    effect_color: Optional[RGBA] = None,
    outline_width: int = 2,
    rotation:    float = 0.0,
) -> BBox:
    """
    Renderizza 'text' sul layer RGBA (H,W,4) a partire dall'ancora (x, y).
    Supporta multilinea (\\n), opacità, bold/italic e 5 effetti WordArt.
    Restituisce il bounding box (x, y, w, h) dell'area modificata.
    """
    if not text:
        return (0, 0, 0, 0)

    patch, ax, ay = render_text_patch(
        text=text,
        font_family=font_family,
        font_size=font_size,
        bold=bold,
        italic=italic,
        color_rgba=color_rgba,
        rotation=rotation,
        opacity=opacity,
        effect=effect,
        effect_color=effect_color,
        outline_width=outline_width,
    )
    px, py = x - ax, y - ay
    _alpha_composite(data, patch, px, py)
    ph, pw = patch.shape[:2]
    return (px, py, pw, ph)
