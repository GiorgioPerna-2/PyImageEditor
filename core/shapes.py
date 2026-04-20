"""
core/shapes.py
--------------
Forme vettoriali: definizione, geometria, handle di manipolazione, rendering.

Ogni Shape è un dataclass mutabile; viene visualizzata come overlay sul canvas
e applicata al layer come patch RGBA tramite render_shape_patch().

Forme supportate
----------------
  ellipse   — cerchio/ellisse (handles: N/S/E/W + angolo rotazione)
  rect      — rettangolo con angoli arrotondabili
  polygon   — poligono regolare N lati
  star      — stella N punte (handle raggio esterno + interno)
  squircle  — superellisse (handle esponente)
  arrow     — freccia direzionale (handle coda/testa/larghezza)
  svg       — SVG caricato dall'utente (handle: scala + rotazione)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Optional

import numpy as np

from PyQt6.QtCore import Qt, QPointF, QRectF, QSizeF
from PyQt6.QtGui import (
    QColor, QPainter, QPainterPath, QPen, QBrush,
    QImage, QTransform, QLinearGradient,
)


# ── Enumerazioni ─────────────────────────────────────────────────────────────

class ShapeKind(Enum):
    ELLIPSE  = auto()
    RECT     = auto()
    POLYGON  = auto()
    STAR     = auto()
    SQUIRCLE = auto()
    ARROW    = auto()
    SVG      = auto()


class HandleID(Enum):
    MOVE          = auto()   # centro — sposta la forma
    ROTATE        = auto()   # sopra il centro — ruota
    RESIZE_E      = auto()
    RESIZE_W      = auto()
    RESIZE_N      = auto()
    RESIZE_S      = auto()
    RESIZE_NE     = auto()
    RESIZE_NW     = auto()
    RESIZE_SE     = auto()
    RESIZE_SW     = auto()
    INNER_RADIUS  = auto()   # stella — raggio interno
    N_SIDES       = auto()   # poligono — cambia N lati (trascina orizzontale)
    SQUIRCLE_EXP  = auto()   # squircle — esponente
    ARROW_TAIL    = auto()
    ARROW_HEAD    = auto()
    ARROW_WIDTH   = auto()   # larghezza trasversale
    SVG_SCALE     = auto()


# ── Stile ────────────────────────────────────────────────────────────────────

@dataclass
class ShapeStyle:
    fill_color:   tuple[int,int,int,int] = (100, 149, 237, 200)  # RGBA
    stroke_color: tuple[int,int,int,int] = (30,  30,  30,  255)
    stroke_width: int   = 2
    opacity:      float = 1.0       # 0‒1, scala entrambi fill e stroke


# ── Handle ───────────────────────────────────────────────────────────────────

@dataclass
class Handle:
    id:       HandleID
    pos:      QPointF          # posizione in coordinate immagine/canvas
    cursor:   Qt.CursorShape = Qt.CursorShape.SizeAllCursor
    visible:  bool = True


# ── Shape dataclass ──────────────────────────────────────────────────────────

@dataclass
class Shape:
    kind:        ShapeKind
    # Bounding box (centro + semiassi in px immagine)
    cx:          float = 200.0
    cy:          float = 200.0
    rx:          float = 80.0    # semiasse orizzontale (anche per arrow/polygon)
    ry:          float = 60.0    # semiasse verticale
    rotation:    float = 0.0     # gradi, senso orario
    style:       ShapeStyle = field(default_factory=ShapeStyle)
    # Params specifici per tipo
    n_sides:     int   = 6       # polygon: numero lati / star: numero punte
    inner_ratio: float = 0.45    # star: raggio_interno / raggio_esterno
    sq_exp:      float = 4.0     # squircle: esponente superellisse
    corner_r:    float = 0.0     # rect: arrotondamento angoli (px)
    # Arrow
    arrow_ax:    float = 0.0     # coda x
    arrow_ay:    float = 0.0     # coda y
    arrow_bx:    float = 0.0     # punta x
    arrow_by:    float = 0.0     # punta y
    arrow_width: float = 20.0    # spessore gambo
    arrow_head_ratio: float = 0.3  # lunghezza testa / lunghezza totale
    # SVG
    svg_path:    str   = ""
    svg_scale:   float = 1.0
    svg_tint:    bool  = False   # se True applica il fill_color come tinta

    # ── Geometria helper ──────────────────────────────────────────────

    def _transform(self) -> QTransform:
        t = QTransform()
        t.translate(self.cx, self.cy)
        t.rotate(self.rotation)
        return t

    def _bbox_rect(self) -> QRectF:
        return QRectF(self.cx - self.rx, self.cy - self.ry,
                      self.rx * 2,       self.ry * 2)

    # ── QPainterPath ──────────────────────────────────────────────────

    def painter_path(self) -> QPainterPath:
        """Ritorna il path nel sistema di coordinate immagine (ruotato)."""
        p = self._local_path()
        t = self._transform()
        return t.map(p)

    def _local_path(self) -> QPainterPath:
        """Path centrato in (0,0) prima della rotazione."""
        k = self.kind
        if k == ShapeKind.ELLIPSE:
            p = QPainterPath()
            p.addEllipse(QRectF(-self.rx, -self.ry, self.rx*2, self.ry*2))
            return p
        if k == ShapeKind.RECT:
            p = QPainterPath()
            r = min(self.corner_r, min(self.rx, self.ry))
            p.addRoundedRect(QRectF(-self.rx, -self.ry, self.rx*2, self.ry*2), r, r)
            return p
        if k == ShapeKind.POLYGON:
            return _regular_polygon(self.rx, self.ry, max(3, self.n_sides))
        if k == ShapeKind.STAR:
            return _star_path(self.rx, self.ry,
                              max(3, self.n_sides), self.inner_ratio)
        if k == ShapeKind.SQUIRCLE:
            return _squircle_path(self.rx, self.ry, max(2.0, self.sq_exp))
        if k == ShapeKind.ARROW:
            return _arrow_path(
                self.arrow_ax - self.cx, self.arrow_ay - self.cy,
                self.arrow_bx - self.cx, self.arrow_by - self.cy,
                self.arrow_width, self.arrow_head_ratio)
        if k == ShapeKind.SVG:
            p = QPainterPath()
            p.addRect(QRectF(-self.rx, -self.ry, self.rx*2, self.ry*2))
            return p
        return QPainterPath()

    # ── Handles ───────────────────────────────────────────────────────

    def handles(self) -> list[Handle]:
        t = self._transform()
        k = self.kind

        def pt(lx, ly):
            return t.map(QPointF(lx, ly))

        hs: list[Handle] = [
            Handle(HandleID.MOVE,     pt(0, 0),          Qt.CursorShape.SizeAllCursor),
            Handle(HandleID.ROTATE,   pt(0, -self.ry - 24), Qt.CursorShape.CrossCursor),
        ]

        if k == ShapeKind.ARROW:
            hs += [
                Handle(HandleID.ARROW_TAIL,  QPointF(self.arrow_ax, self.arrow_ay),
                       Qt.CursorShape.SizeAllCursor),
                Handle(HandleID.ARROW_HEAD,  QPointF(self.arrow_bx, self.arrow_by),
                       Qt.CursorShape.SizeAllCursor),
                Handle(HandleID.ARROW_WIDTH,
                       QPointF((self.arrow_ax+self.arrow_bx)/2 + self.arrow_width,
                                (self.arrow_ay+self.arrow_by)/2),
                       Qt.CursorShape.SizeHorCursor),
            ]
            return hs

        # Resize handles sul bounding box
        hs += [
            Handle(HandleID.RESIZE_E,  pt( self.rx,        0),          Qt.CursorShape.SizeHorCursor),
            Handle(HandleID.RESIZE_W,  pt(-self.rx,        0),          Qt.CursorShape.SizeHorCursor),
            Handle(HandleID.RESIZE_N,  pt(0,         -self.ry),         Qt.CursorShape.SizeVerCursor),
            Handle(HandleID.RESIZE_S,  pt(0,          self.ry),         Qt.CursorShape.SizeVerCursor),
            Handle(HandleID.RESIZE_NE, pt( self.rx,  -self.ry),         Qt.CursorShape.SizeBDiagCursor),
            Handle(HandleID.RESIZE_NW, pt(-self.rx,  -self.ry),         Qt.CursorShape.SizeFDiagCursor),
            Handle(HandleID.RESIZE_SE, pt( self.rx,   self.ry),         Qt.CursorShape.SizeFDiagCursor),
            Handle(HandleID.RESIZE_SW, pt(-self.rx,   self.ry),         Qt.CursorShape.SizeBDiagCursor),
        ]
        if k == ShapeKind.STAR:
            hs.append(Handle(HandleID.INNER_RADIUS,
                             pt(self.rx * self.inner_ratio * 0.7,
                                -self.ry * self.inner_ratio * 0.7),
                             Qt.CursorShape.SizeAllCursor))
        if k == ShapeKind.SQUIRCLE:
            hs.append(Handle(HandleID.SQUIRCLE_EXP,
                             pt(self.rx * 0.6, -self.ry * 0.6),
                             Qt.CursorShape.SizeHorCursor))
        if k in (ShapeKind.POLYGON, ShapeKind.STAR):
            hs.append(Handle(HandleID.N_SIDES,
                             pt(-self.rx - 18, -self.ry - 18),
                             Qt.CursorShape.SizeHorCursor))
        if k == ShapeKind.SVG:
            hs.append(Handle(HandleID.SVG_SCALE,
                             pt(self.rx, self.ry),
                             Qt.CursorShape.SizeFDiagCursor))
        return hs

    def apply_handle_drag(self, hid: HandleID,
                           dx: float, dy: float,
                           vp_to_scene_scale: float = 1.0) -> None:
        """
        Muove un handle di (dx,dy) in coordinate immagine.
        vp_to_scene_scale = 1/zoom, utile per i controlli discreti.
        """
        k = self.kind
        MIN_R = 4.0

        if hid == HandleID.MOVE:
            self.cx += dx; self.cy += dy
            if k == ShapeKind.ARROW:
                self.arrow_ax += dx; self.arrow_ay += dy
                self.arrow_bx += dx; self.arrow_by += dy

        elif hid == HandleID.ROTATE:
            # angolo dal centro al punto corrente dell'handle
            handle_scene = self._transform().map(QPointF(0, -self.ry - 24))
            new_angle = math.degrees(math.atan2(
                handle_scene.y() + dy - self.cy,
                handle_scene.x() + dx - self.cx)) + 90
            self.rotation = new_angle % 360

        elif hid == HandleID.RESIZE_E:
            self.rx = max(MIN_R, self.rx + dx)
        elif hid == HandleID.RESIZE_W:
            self.rx = max(MIN_R, self.rx - dx); self.cx += dx
        elif hid == HandleID.RESIZE_N:
            self.ry = max(MIN_R, self.ry - dy); self.cy += dy
        elif hid == HandleID.RESIZE_S:
            self.ry = max(MIN_R, self.ry + dy)
        elif hid == HandleID.RESIZE_NE:
            self.rx = max(MIN_R, self.rx + dx)
            self.ry = max(MIN_R, self.ry - dy); self.cy += dy
        elif hid == HandleID.RESIZE_NW:
            self.rx = max(MIN_R, self.rx - dx); self.cx += dx
            self.ry = max(MIN_R, self.ry - dy); self.cy += dy
        elif hid == HandleID.RESIZE_SE:
            self.rx = max(MIN_R, self.rx + dx)
            self.ry = max(MIN_R, self.ry + dy)
        elif hid == HandleID.RESIZE_SW:
            self.rx = max(MIN_R, self.rx - dx); self.cx += dx
            self.ry = max(MIN_R, self.ry + dy)

        elif hid == HandleID.INNER_RADIUS:
            d = math.hypot(dx, dy) * (1 if dx + dy > 0 else -1)
            self.inner_ratio = max(0.05, min(0.95, self.inner_ratio + d / max(self.rx, 1)))

        elif hid == HandleID.SQUIRCLE_EXP:
            self.sq_exp = max(2.0, min(20.0, self.sq_exp + dx * 0.05))

        elif hid == HandleID.N_SIDES:
            if abs(dx) > 10:
                self.n_sides = max(3, min(20, self.n_sides + (1 if dx > 0 else -1)))

        elif hid == HandleID.ARROW_TAIL:
            self.arrow_ax += dx; self.arrow_ay += dy
            self._sync_arrow_center()
        elif hid == HandleID.ARROW_HEAD:
            self.arrow_bx += dx; self.arrow_by += dy
            self._sync_arrow_center()
        elif hid == HandleID.ARROW_WIDTH:
            self.arrow_width = max(4.0, self.arrow_width + dx)

        elif hid == HandleID.SVG_SCALE:
            d = (dx + dy) / max(self.rx + self.ry, 1) * 0.5
            self.svg_scale = max(0.05, self.svg_scale + d)
            self.rx = max(MIN_R, self.rx + dx)
            self.ry = max(MIN_R, self.ry + dy)

    def _sync_arrow_center(self) -> None:
        self.cx = (self.arrow_ax + self.arrow_bx) / 2
        self.cy = (self.arrow_ay + self.arrow_by) / 2

    # ── Rendering ─────────────────────────────────────────────────────

    def render_patch(self, canvas_w: int, canvas_h: int) -> tuple[np.ndarray, int, int]:
        """
        Renderizza la forma in un patch RGBA numpy e restituisce
        (patch_array, offset_x, offset_y) nel sistema coordinate canvas.
        """
        # Calcola bounding box della forma ruotata + margine per lo stroke
        path  = self.painter_path()
        brect = path.boundingRect()
        margin = max(self.style.stroke_width + 4, 10)

        # offset del patch nel canvas
        ox = max(0, int(brect.left()) - margin)
        oy = max(0, int(brect.top())  - margin)
        ox2 = min(canvas_w, int(brect.right())  + margin)
        oy2 = min(canvas_h, int(brect.bottom()) + margin)
        if ox >= ox2 or oy >= oy2:
            return np.zeros((1, 1, 4), dtype=np.uint8), 0, 0

        pw = ox2 - ox
        ph = oy2 - oy

        img = QImage(pw, ph, QImage.Format.Format_RGBA8888)
        img.fill(Qt.GlobalColor.transparent)
        painter = QPainter(img)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.translate(-ox, -oy)

        self._paint_on(painter, canvas_w, canvas_h)
        painter.end()

        ptr = img.bits(); ptr.setsize(img.sizeInBytes())
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape((ph, pw, 4)).copy()
        return arr, ox, oy

    def _paint_on(self, painter: QPainter,
                  canvas_w: int = 0, canvas_h: int = 0) -> None:
        """
        Disegna la forma sul painter corrente (coordinate canvas).
        Usata sia da render_patch() che dall'anteprima canvas overlay.
        """
        op = max(0.0, min(1.0, self.style.opacity))
        fc = self.style.fill_color
        sc = self.style.stroke_color
        fill   = QColor(fc[0], fc[1], fc[2], int(fc[3] * op))
        stroke = QColor(sc[0], sc[1], sc[2], int(sc[3] * op))

        pen = QPen(stroke, self.style.stroke_width)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        if self.style.stroke_width == 0:
            pen = Qt.PenStyle.NoPen  # type: ignore[assignment]

        painter.setPen(pen)
        painter.setBrush(QBrush(fill))

        if self.kind == ShapeKind.SVG and self.svg_path:
            self._paint_svg(painter)
        else:
            painter.drawPath(self.painter_path())

    def _paint_svg(self, painter: QPainter) -> None:
        try:
            from PyQt6.QtSvg import QSvgRenderer
        except ImportError:
            painter.drawRect(QRectF(self.cx - self.rx, self.cy - self.ry,
                                    self.rx * 2, self.ry * 2))
            return

        renderer = QSvgRenderer(self.svg_path)
        if not renderer.isValid():
            return

        w = max(1, int(self.rx * 2) + 2)
        h = max(1, int(self.ry * 2) + 2)

        painter.save()
        painter.translate(self.cx, self.cy)
        painter.rotate(self.rotation)
        painter.setOpacity(self.style.opacity)

        target = QRectF(-self.rx, -self.ry, self.rx * 2, self.ry * 2)

        if self.svg_tint:
            # Renderizza SVG su buffer trasparente
            from PyQt6.QtGui import QImage
            tmp = QImage(w, h, QImage.Format.Format_RGBA8888)
            tmp.fill(0)
            p2 = QPainter(tmp)
            p2.setRenderHint(QPainter.RenderHint.Antialiasing)
            renderer.render(p2, QRectF(0, 0, w, h))
            p2.end()

            # SourceAtop: colora SOLO i pixel non-trasparenti dell'SVG
            # result = fill_color * dst.alpha   →   sfondo resta trasparente
            fc = self.style.fill_color
            p3 = QPainter(tmp)
            p3.setCompositionMode(
                QPainter.CompositionMode.CompositionMode_SourceAtop)
            p3.fillRect(0, 0, w, h,
                        QColor(fc[0], fc[1], fc[2], fc[3]))
            p3.end()

            painter.drawImage(target, tmp)
        else:
            renderer.render(painter, target)

        painter.restore()


# ── Rendering su layer (alpha-composite) ──────────────────────────────────────

def render_shape_onto(
    data:     np.ndarray,
    shape:    Shape,
) -> tuple[int, int, int, int]:
    """
    Renderizza shape su data (RGBA uint8, H×W×4) in-place.
    Restituisce il DirtyRect (x0,y0,x1,y1) in coordinate layer.
    """
    h, w = data.shape[:2]
    patch, ox, oy = shape.render_patch(w, h)
    ph, pw = patch.shape[:2]
    x1, y1 = ox + pw, oy + ph
    x1 = min(w, x1); y1 = min(h, y1)
    pw2, ph2 = x1 - ox, y1 - oy
    if pw2 <= 0 or ph2 <= 0:
        return (0, 0, 0, 0)
    # Se il patch è interamente trasparente (forma OOB o opacity=0) non toccare il layer
    if patch[:ph2, :pw2, 3].max() == 0:
        return (0, 0, 0, 0)

    s   = patch[:ph2, :pw2].astype(np.float32)
    d   = data[oy:y1, ox:x1].astype(np.float32)
    sa  = s[:, :, 3:4] / 255.0
    da  = d[:, :, 3:4] / 255.0
    oa  = sa + da * (1.0 - sa)
    den = np.maximum(oa, 1e-8)
    out_rgb = (s[:, :, :3] * sa + d[:, :, :3] * da * (1.0 - sa)) / den
    out = np.empty_like(d)
    out[:, :, :3] = out_rgb
    out[:, :, 3:4] = oa * 255.0
    data[oy:y1, ox:x1] = np.clip(out, 0, 255).astype(np.uint8)
    return (ox, oy, x1, y1)


# ── Factory ───────────────────────────────────────────────────────────────────

def default_shape(kind: ShapeKind, cx: float, cy: float) -> Shape:
    """Crea una forma con valori di default centrata in (cx,cy)."""
    s = Shape(kind=kind, cx=cx, cy=cy)
    if kind == ShapeKind.ARROW:
        s.arrow_ax = cx - 80; s.arrow_ay = cy
        s.arrow_bx = cx + 80; s.arrow_by = cy
        s.arrow_width = 22
        s._sync_arrow_center()
    return s


# ── Geometrie locali (helpers interni) ───────────────────────────────────────

def _regular_polygon(rx: float, ry: float, n: int) -> QPainterPath:
    p = QPainterPath()
    angle0 = -math.pi / 2
    pts = [QPointF(rx * math.cos(angle0 + 2*math.pi*i/n),
                   ry * math.sin(angle0 + 2*math.pi*i/n)) for i in range(n)]
    p.moveTo(pts[0])
    for pt in pts[1:]:
        p.lineTo(pt)
    p.closeSubpath()
    return p


def _star_path(rx: float, ry: float, n: int, inner_ratio: float) -> QPainterPath:
    p = QPainterPath()
    angle0 = -math.pi / 2
    pts: list[QPointF] = []
    for i in range(n):
        outer = QPointF(rx * math.cos(angle0 + 2*math.pi*i/n),
                        ry * math.sin(angle0 + 2*math.pi*i/n))
        inner = QPointF(rx * inner_ratio * math.cos(angle0 + 2*math.pi*(i+0.5)/n),
                        ry * inner_ratio * math.sin(angle0 + 2*math.pi*(i+0.5)/n))
        pts += [outer, inner]
    p.moveTo(pts[0])
    for pt in pts[1:]:
        p.lineTo(pt)
    p.closeSubpath()
    return p


def _squircle_path(rx: float, ry: float, exp: float, n_pts: int = 128) -> QPainterPath:
    """Superellisse: |x/rx|^exp + |y/ry|^exp = 1."""
    p = QPainterPath()
    inv = 1.0 / exp
    angles = [2 * math.pi * i / n_pts for i in range(n_pts)]
    pts = []
    for a in angles:
        ca, sa = math.cos(a), math.sin(a)
        x = math.copysign(abs(ca) ** inv, ca) * rx
        y = math.copysign(abs(sa) ** inv, sa) * ry
        pts.append(QPointF(x, y))
    p.moveTo(pts[0])
    for pt in pts[1:]:
        p.lineTo(pt)
    p.closeSubpath()
    return p


def _arrow_path(ax: float, ay: float, bx: float, by: float,
                width: float, head_ratio: float) -> QPainterPath:
    """Freccia dal punto a al punto b, con gambo di larghezza 'width'."""
    dx, dy = bx - ax, by - ay
    length = math.hypot(dx, dy)
    if length < 1.0:
        return QPainterPath()
    ux, uy = dx / length, dy / length   # versore direzione
    px, py = -uy, ux                     # perpendicolare

    head_len  = length * head_ratio
    stem_len  = length - head_len
    stem_w    = width / 2.0
    head_w    = width            # larghezza massima della punta

    # Punti del poligono (nel sistema locale, poi traslato)
    # gambo: rettangolo
    # punta: triangolo
    p = QPainterPath()
    pts = [
        QPointF(ax + px * stem_w,       ay + py * stem_w),
        QPointF(ax + ux * stem_len + px * stem_w,
                ay + uy * stem_len + py * stem_w),
        QPointF(ax + ux * stem_len + px * head_w,
                ay + uy * stem_len + py * head_w),
        QPointF(bx,  by),
        QPointF(ax + ux * stem_len - px * head_w,
                ay + uy * stem_len - py * head_w),
        QPointF(ax + ux * stem_len - px * stem_w,
                ay + uy * stem_len - py * stem_w),
        QPointF(ax - px * stem_w,       ay - py * stem_w),
    ]
    p.moveTo(pts[0])
    for pt in pts[1:]:
        p.lineTo(pt)
    p.closeSubpath()
    return p
