"""Drawing tool operations on RGBA numpy arrays."""
from __future__ import annotations

from typing import Tuple
import numpy as np

ColorRGBA = Tuple[int, int, int, int]


# ── Brush helpers ────────────────────────────────────────────────────────────

def _brush_mask(r: int, hardness: float) -> np.ndarray:
    """Return a float32 (2r+1, 2r+1) alpha mask for a circular brush dab."""
    size = 2 * r + 1
    yy, xx = np.mgrid[0:size, 0:size]
    dist = np.sqrt(((xx - r) ** 2 + (yy - r) ** 2).astype(np.float32))
    mask = np.zeros_like(dist)
    inner = r * max(0.0, min(1.0, hardness))
    mask[dist <= inner] = 1.0
    soft = (dist > inner) & (dist <= r)
    if soft.any() and r > inner:
        mask[soft] = 1.0 - (dist[soft] - inner) / max(r - inner, 1e-6)
    return mask


def apply_brush(
    array: np.ndarray,
    x: int,
    y: int,
    size: int,
    color: ColorRGBA,
    opacity: float,
    hardness: float = 0.8,
) -> None:
    """Paint a circular brush dab at (x, y) – normal blend."""
    _paint_dab(array, x, y, size, color, opacity, hardness, eraser=False)


def apply_eraser(
    array: np.ndarray,
    x: int,
    y: int,
    size: int,
    opacity: float,
    hardness: float = 0.8,
) -> None:
    """Erase alpha at (x, y)."""
    _paint_dab(array, x, y, size, (0, 0, 0, 0), opacity, hardness, eraser=True)


def _paint_dab(
    array: np.ndarray,
    x: int,
    y: int,
    size: int,
    color: ColorRGBA,
    opacity: float,
    hardness: float,
    eraser: bool,
) -> None:
    h, w = array.shape[:2]
    r = max(1, size // 2)
    full = _brush_mask(r, hardness)

    # Bounding box clipped to image
    y0_src = max(0, -( y - r));  y0_dst = max(0, y - r)
    y1_dst = min(h, y + r + 1);  y1_src = y0_src + (y1_dst - y0_dst)
    x0_src = max(0, -(x - r));   x0_dst = max(0, x - r)
    x1_dst = min(w, x + r + 1);  x1_src = x0_src + (x1_dst - x0_dst)

    if y0_dst >= y1_dst or x0_dst >= x1_dst:
        return

    mask = full[y0_src:y1_src, x0_src:x1_src] * float(opacity)
    region = array[y0_dst:y1_dst, x0_dst:x1_dst].astype(np.float32)

    if eraser:
        region[:, :, 3] *= 1.0 - mask
    else:
        cr, cg, cb, ca = color
        src_a = mask * (ca / 255.0)
        dst_a = region[:, :, 3] / 255.0
        out_a = src_a + dst_a * (1.0 - src_a)
        safe = np.where(out_a > 0, out_a, 1.0)
        for c, cv in enumerate((cr, cg, cb)):
            region[:, :, c] = (cv * src_a + region[:, :, c] * dst_a * (1.0 - src_a)) / safe
        region[:, :, 3] = out_a * 255.0

    array[y0_dst:y1_dst, x0_dst:x1_dst] = np.clip(region, 0, 255).astype(np.uint8)


# ── Smudge ───────────────────────────────────────────────────────────────────

def apply_smudge(
    array: np.ndarray,
    x: int,
    y: int,
    size: int,
    strength: float = 0.5,
) -> None:
    """Blur the local area around (x, y) – smudge effect."""
    import cv2
    h, w = array.shape[:2]
    r = max(2, size // 2)
    y1 = max(0, y - r); y2 = min(h, y + r + 1)
    x1 = max(0, x - r); x2 = min(w, x + r + 1)
    if y1 >= y2 or x1 >= x2:
        return
    region = array[y1:y2, x1:x2].copy()
    ksize = max(3, (r // 2) * 2 + 1)
    blurred = cv2.GaussianBlur(region, (ksize, ksize), 0)
    mixed = cv2.addWeighted(region, 1.0 - strength, blurred, float(strength), 0)
    array[y1:y2, x1:x2] = mixed


# ── Flood fill ───────────────────────────────────────────────────────────────

def flood_fill(
    array: np.ndarray,
    x: int,
    y: int,
    fill_color: ColorRGBA,
    tolerance: int = 32,
) -> None:
    """BFS flood-fill starting at (x, y)."""
    from collections import deque
    h, w = array.shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        return

    target = array[y, x].astype(np.int32)
    fill = np.array(fill_color, dtype=np.uint8)

    if np.array_equal(array[y, x], fill):
        return

    visited = np.zeros((h, w), dtype=bool)
    queue: deque[tuple[int, int]] = deque()
    queue.append((x, y))
    visited[y, x] = True
    thresh = tolerance * 4

    while queue:
        cx, cy = queue.popleft()
        array[cy, cx] = fill
        for nx_, ny_ in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
            if 0 <= nx_ < w and 0 <= ny_ < h and not visited[ny_, nx_]:
                pixel = array[ny_, nx_].astype(np.int32)
                if int(np.abs(pixel - target).sum()) <= thresh:
                    visited[ny_, nx_] = True
                    queue.append((nx_, ny_))


# ── Line interpolation ────────────────────────────────────────────────────────

def bresenham(x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
    """Return all pixel positions on the line from (x0,y0) to (x1,y1)."""
    pts: list[tuple[int, int]] = []
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    x, y = x0, y0
    while True:
        pts.append((x, y))
        if x == x1 and y == y1:
            break
        e2 = err * 2
        if e2 > -dy:
            err -= dy; x += sx
        if e2 < dx:
            err += dx; y += sy
    return pts


# ── Text rendering ────────────────────────────────────────────────────────────

def render_text(
    array: np.ndarray,
    x: int,
    y: int,
    text: str,
    font_size: int,
    color: ColorRGBA,
) -> None:
    """Render text onto the RGBA array at (x, y) using Pillow."""
    from PIL import Image, ImageDraw, ImageFont

    pil_img = Image.fromarray(array, mode="RGBA")
    draw = ImageDraw.Draw(pil_img)

    font: ImageFont.FreeTypeFont | ImageFont.ImageFont
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ):
        try:
            font = ImageFont.truetype(path, font_size)
            break
        except (IOError, OSError):
            continue
    else:
        font = ImageFont.load_default()

    draw.text((x, y), text, fill=color, font=font)
    array[:] = np.array(pil_img)


# ── Gradiente ─────────────────────────────────────────────────────────────────

def apply_gradient(
    array: np.ndarray,
    x1: float, y1: float,
    x2: float, y2: float,
    color1: tuple[int, int, int, int],
    color2: tuple[int, int, int, int],
    mode: str = "linear",          # "linear" | "radial"
    opacity: float = 1.0,          # 0.0–1.0, scala entrambi i colori
    selection_mask: np.ndarray | None = None,
    layer_offset_x: int = 0,
    layer_offset_y: int = 0,
) -> None:
    """
    Applica un gradiente sull'intera area del layer (o solo nella selezione).
    x1,y1 → x2,y2 sono in coordinate canvas.
    opacity scala l'alpha di entrambi i colori (0=trasparente, 1=pieno).
    """
    h, w = array.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # Converti coordinate canvas → layer
    lx1 = x1 - layer_offset_x;  ly1 = y1 - layer_offset_y
    lx2 = x2 - layer_offset_x;  ly2 = y2 - layer_offset_y

    if mode == "radial":
        radius = max(1.0, np.hypot(lx2 - lx1, ly2 - ly1))
        dist = np.sqrt((xx - lx1) ** 2 + (yy - ly1) ** 2)
        t = np.clip(dist / radius, 0.0, 1.0)
    else:
        dx_v = lx2 - lx1;  dy_v = ly2 - ly1
        length_sq = dx_v ** 2 + dy_v ** 2
        if length_sq < 1.0:
            return
        t = ((xx - lx1) * dx_v + (yy - ly1) * dy_v) / length_sq
        t = np.clip(t, 0.0, 1.0)

    c1 = np.array(color1, dtype=np.float32)
    c2 = np.array(color2, dtype=np.float32)
    gradient = (c1 * (1.0 - t[:, :, np.newaxis]) +
                c2 *        t[:, :, np.newaxis])

    src_a = gradient[:, :, 3:4] / 255.0 * opacity

    # Applica selezione canvas → layer
    if selection_mask is not None:
        mh, mw = selection_mask.shape
        oy0 = max(0, layer_offset_y);  oy1 = min(mh, layer_offset_y + h)
        ox0 = max(0, layer_offset_x);  ox1 = min(mw, layer_offset_x + w)
        sel = np.zeros((h, w), dtype=np.float32)
        if oy1 > oy0 and ox1 > ox0:
            ly0 = oy0 - layer_offset_y;  lx0s = ox0 - layer_offset_x
            sel[ly0:ly0+(oy1-oy0), lx0s:lx0s+(ox1-ox0)] = \
                selection_mask[oy0:oy1, ox0:ox1].astype(np.float32)
        src_a = src_a * sel[:, :, np.newaxis]

    dst_f = array.astype(np.float32)
    dst_a = dst_f[:, :, 3:4] / 255.0
    out_a = src_a + dst_a * (1.0 - src_a)
    denom = np.maximum(out_a, 1e-8)
    out_rgb = (gradient[:, :, :3] * src_a +
               dst_f[:, :, :3] * dst_a * (1.0 - src_a)) / denom
    result = dst_f.copy()
    result[:, :, :3] = out_rgb
    result[:, :, 3:4] = out_a * 255.0
    array[:] = np.clip(result, 0, 255).astype(np.uint8)


# ── Healing brush ─────────────────────────────────────────────────────────────

def build_heal_mask(
    array: np.ndarray,
    cx: float, cy: float,
    size: int,
    hardness: float = 0.8,
) -> np.ndarray:
    """Costruisce la maschera uint8 per heal (stesso profilo circolare del pennello)."""
    h, w = array.shape[:2]
    r = max(1, size // 2)
    diam = 2 * r + 1
    yy, xx = np.mgrid[:diam, :diam]
    dist = np.sqrt(((xx - r) ** 2 + (yy - r) ** 2).astype(np.float32))
    inner = r * max(0.0, min(1.0, hardness))
    m = np.zeros(dist.shape, dtype=np.float32)
    m[dist <= inner] = 1.0
    soft = (dist > inner) & (dist <= r)
    if soft.any() and r > inner:
        m[soft] = 1.0 - (dist[soft] - inner) / max(r - inner, 1e-6)

    mask = np.zeros((h, w), dtype=np.uint8)
    icx, icy = int(round(cx)), int(round(cy))
    iy0 = max(0, icy - r);  iy1 = min(h, icy + r + 1)
    ix0 = max(0, icx - r);  ix1 = min(w, icx + r + 1)
    if iy0 >= iy1 or ix0 >= ix1:
        return mask
    my0 = iy0 - (icy - r)
    mx0 = ix0 - (icx - r)
    rh = iy1 - iy0
    rw = ix1 - ix0
    region = (m[my0:my0 + rh, mx0:mx0 + rw] * 255).astype(np.uint8)
    np.maximum(mask[iy0:iy1, ix0:ix1], region, out=mask[iy0:iy1, ix0:ix1])
    return mask


def apply_healing(
    array: np.ndarray,
    heal_mask: np.ndarray,
    radius: int = 5,
) -> None:
    """cv2.inpaint sull'area coperta da heal_mask. Modifica array in-place."""
    import cv2
    if not heal_mask.any():
        return
    bgr = cv2.cvtColor(array[:, :, :3], cv2.COLOR_RGB2BGR)
    healed_bgr = cv2.inpaint(bgr, heal_mask, radius, cv2.INPAINT_TELEA)
    array[:, :, :3] = cv2.cvtColor(healed_bgr, cv2.COLOR_BGR2RGB)
