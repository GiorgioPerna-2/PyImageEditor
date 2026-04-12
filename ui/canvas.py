from __future__ import annotations

from enum import Enum, auto
from typing import Optional

import numpy as np
from PyQt6.QtCore import Qt, QRectF, QPointF, pyqtSignal
from PyQt6.QtGui import (
    QImage, QPixmap, QWheelEvent, QMouseEvent, QKeyEvent,
    QPainter, QPen, QColor, QBrush, QFont, QFontMetrics,
)
from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem

MIN_ZOOM = 0.05
MAX_ZOOM = 32.0
ZOOM_STEP = 1.15
HANDLE_RADIUS = 6
HANDLE_HIT = 20
MIN_CROP_PX = 10


class CanvasMode(Enum):
    NORMAL     = auto()
    CROP       = auto()
    MOVE       = auto()
    BRUSH      = auto()
    ERASER     = auto()
    SMUDGE     = auto()
    EYEDROPPER = auto()
    TEXT       = auto()
    GRABCUT    = auto()   # rettangolo selezione per GrabCut
    SELECT_RECT   = auto()   # selezione rettangolare
    SELECT_ELLIPSE = auto()  # selezione ellittica
    SELECT_LASSO  = auto()   # lasso libero
    CLONE         = auto()   # timbro clone


class _Handle(Enum):
    NONE = auto(); TOP_LEFT = auto(); TOP = auto(); TOP_RIGHT = auto()
    RIGHT = auto(); BOTTOM_RIGHT = auto(); BOTTOM = auto()
    BOTTOM_LEFT = auto(); LEFT = auto(); MOVE = auto()


_HANDLE_CURSORS: dict[_Handle, Qt.CursorShape] = {
    _Handle.TOP_LEFT:     Qt.CursorShape.SizeFDiagCursor,
    _Handle.TOP_RIGHT:    Qt.CursorShape.SizeBDiagCursor,
    _Handle.BOTTOM_LEFT:  Qt.CursorShape.SizeBDiagCursor,
    _Handle.BOTTOM_RIGHT: Qt.CursorShape.SizeFDiagCursor,
    _Handle.TOP:          Qt.CursorShape.SizeVerCursor,
    _Handle.BOTTOM:       Qt.CursorShape.SizeVerCursor,
    _Handle.LEFT:         Qt.CursorShape.SizeHorCursor,
    _Handle.RIGHT:        Qt.CursorShape.SizeHorCursor,
    _Handle.MOVE:         Qt.CursorShape.SizeAllCursor,
    _Handle.NONE:         Qt.CursorShape.ArrowCursor,
}

# Radius (in viewport px) within which a click on an existing text position
# triggers a drag instead of placing a new position
_TEXT_DRAG_HIT = 30


class ImageCanvas(QGraphicsView):
    zoom_changed    = pyqtSignal(float)
    pixel_hovered   = pyqtSignal(int, int)
    canvas_cleared  = pyqtSignal()
    crop_committed  = pyqtSignal(int, int, int, int)
    grabcut_rect_committed = pyqtSignal(int, int, int, int)   # x,y,w,h per GrabCut
    layer_moved     = pyqtSignal(int, int)

    # Selezione
    selection_rect_ready   = pyqtSignal(int, int, int, int)    # x,y,w,h rect finale
    selection_ellipse_ready = pyqtSignal(int, int, int, int)   # x,y,w,h rect bounding
    selection_lasso_ready  = pyqtSignal(object)                # list[(x,y)]
    selection_cancelled    = pyqtSignal()

    # Clone stamp
    clone_source_set   = pyqtSignal(float, float)              # scene x,y sorgente
    clone_stroke_begin = pyqtSignal(float, float)              # dst x,y
    clone_stroke_move  = pyqtSignal(float, float)              # dst x,y
    clone_stroke_end   = pyqtSignal()

    # Drawing strokes (scene/image coordinates)
    stroke_begin = pyqtSignal(float, float)
    stroke_move  = pyqtSignal(float, float)
    stroke_end   = pyqtSignal()

    # Eyedropper: clicked pixel in image coordinates
    eyedropper_pick = pyqtSignal(int, int)

    # Text tool: current text anchor in scene (=image) coordinates
    text_anchor_changed = pyqtSignal(float, float)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self._pixmap_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)
        self.setScene(self._scene)

        self._zoom_factor: float = 1.0
        self._panning = False
        self._space_held = False
        self._last_pan_pos: Optional[QPointF] = None
        self._image_array: Optional[np.ndarray] = None

        # Crop state
        self._mode = CanvasMode.NORMAL
        self._crop_rect: Optional[QRectF] = None
        self._active_handle = _Handle.NONE
        self._drag_start_pos: Optional[QPointF] = None
        self._drag_start_rect: Optional[QRectF] = None

        # Move layer state
        self._move_origin: Optional[QPointF] = None

        # Drawing state (brush/eraser/smudge)
        self._drawing = False
        self._brush_size: int = 20
        self._cursor_pos: Optional[QPointF] = None   # viewport px

        # Text state
        self._text_anchor: Optional[QPointF] = None  # scene coords
        self._text_dragging = False
        self._text_preview: dict = {}   # text, font_family, font_size, bold, italic, color

        # Selezione stato
        self._sel_start: Optional[QPointF] = None      # scene coords, primo click
        self._sel_rect: Optional[QRectF] = None        # rettangolo corrente (scena)
        self._sel_lasso_pts: list[tuple[int,int]] = [] # punti lasso (scene ints)
        # Maschera di selezione corrente (per overlay) — aggiornata dal main_window
        self._sel_mask: Optional[object] = None        # np.ndarray bool
        self._sel_outline: list[tuple[int,int,int,int]] = []   # segmenti marching ants
        self._ants_offset: int = 0                     # fase animazione

        # Clone stamp stato
        self._clone_source: Optional[QPointF] = None   # scene coords sorgente
        self._clone_drawing = False
        self._clone_alt_held = False

        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setBackgroundBrush(Qt.GlobalColor.darkGray)
        self.setFrameShape(self.Shape.NoFrame)
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)
        self.setAcceptDrops(True)
        # NON impostare setAcceptDrops sul viewport: QGraphicsView forwardea
        # automaticamente i drag event dal viewport alla view tramite viewportEvent.
        # Impostarlo su entrambi causa conflitti.

    # ── Segnali drag & drop ───────────────────────────────────────────

    file_dropped  = pyqtSignal(str)     # path locale
    image_dropped = pyqtSignal(object)  # np.ndarray RGBA

    # ── API drag & drop ───────────────────────────────────────────────

    def dragEnterEvent(self, event) -> None:
        mime = event.mimeData()
        if mime.hasUrls() or mime.hasImage():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event) -> None:
        """Deve accettare esplicitamente ad ogni movimento, altrimenti Qt
        revoca il permesso di drop prima che avvenga."""
        mime = event.mimeData()
        if mime.hasUrls() or mime.hasImage():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event) -> None:
        mime = event.mimeData()

        if mime.hasUrls():
            for url in mime.urls():
                path = url.toLocalFile()
                if path:
                    self.file_dropped.emit(path)
                    event.acceptProposedAction()
                    return
            event.ignore()
            return

        if mime.hasImage():
            qimg = mime.imageData()
            if qimg is not None and not qimg.isNull():
                qimg = qimg.convertToFormat(QImage.Format.Format_RGBA8888)
                w, h = qimg.width(), qimg.height()
                bpl = qimg.bytesPerLine()
                ptr = qimg.constBits()
                arr = np.frombuffer(memoryview(ptr), dtype=np.uint8)
                arr = arr.reshape(h, bpl // 4, 4)[:, :w, :].copy()
                self.image_dropped.emit(arr)
                event.acceptProposedAction()
                return

        event.ignore()
    # ── API immagine ──────────────────────────────────────────────────

    def set_image(self, array: np.ndarray) -> None:
        self._image_array = array
        pixmap = self._to_pixmap(array)
        self._pixmap_item.setPixmap(pixmap)
        self._scene.setSceneRect(QRectF(pixmap.rect()))
        self.fit_in_view()

    def update_image(self, array: np.ndarray) -> None:
        self._image_array = array
        self._pixmap_item.setPixmap(self._to_pixmap(array))

    def get_image(self) -> Optional[np.ndarray]:
        return self._image_array

    def clear(self) -> None:
        self.end_crop_mode(confirm=False)
        self._mode = CanvasMode.NORMAL
        self._text_anchor = None
        self._pixmap_item.setPixmap(QPixmap())
        self._image_array = None
        self.canvas_cleared.emit()

    # ── API zoom ──────────────────────────────────────────────────────

    def fit_in_view(self) -> None:
        r = self._pixmap_item.boundingRect()
        if not r.isNull():
            self.fitInView(r, Qt.AspectRatioMode.KeepAspectRatio)
            self._zoom_factor = self.transform().m11()
            self.zoom_changed.emit(self._zoom_factor)

    def set_zoom(self, f: float) -> None:
        f = max(MIN_ZOOM, min(MAX_ZOOM, f))
        self.resetTransform(); self.scale(f, f)
        self._zoom_factor = f; self.zoom_changed.emit(f)

    def zoom_in(self)  -> None: self.set_zoom(self._zoom_factor * ZOOM_STEP)
    def zoom_out(self) -> None: self.set_zoom(self._zoom_factor / ZOOM_STEP)

    @property
    def zoom_factor(self) -> float:
        return self._zoom_factor

    # ── API modalità ──────────────────────────────────────────────────

    def start_crop_mode(self) -> None:
        if self._image_array is None: return
        h, w = self._image_array.shape[:2]
        self._crop_rect = QRectF(0.0, 0.0, float(w), float(h))
        self._mode = CanvasMode.CROP
        self.viewport().update()

    def end_crop_mode(self, confirm: bool) -> None:
        if self._mode != CanvasMode.CROP: return
        if confirm and self._crop_rect and self._image_array is not None:
            h, w = self._image_array.shape[:2]
            r = self._crop_rect.normalized()
            x  = max(0, round(r.left()));   y  = max(0, round(r.top()))
            x2 = min(w, round(r.right()));  y2 = min(h, round(r.bottom()))
            if x2 - x >= MIN_CROP_PX and y2 - y >= MIN_CROP_PX:
                self.crop_committed.emit(x, y, x2 - x, y2 - y)
        self._mode = CanvasMode.NORMAL
        self._crop_rect = None; self._active_handle = _Handle.NONE
        self._drag_start_pos = None; self._drag_start_rect = None
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.viewport().update()

    def start_grabcut_mode(self) -> None:
        """Modalità rettangolo GrabCut: come crop ma non modifica l'immagine."""
        if self._image_array is None: return
        self._crop_rect = None
        self._active_handle = _Handle.NONE
        self._drag_start_pos = None
        self._drag_start_rect = None
        self._mode = CanvasMode.GRABCUT
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.viewport().setFocus()   # serve per ricevere Invio/Esc
        self.viewport().update()

    def end_grabcut_mode(self, confirm: bool) -> None:
        if self._mode != CanvasMode.GRABCUT: return
        if confirm and self._crop_rect and self._image_array is not None:
            h, w = self._image_array.shape[:2]
            r = self._crop_rect.normalized()
            x  = max(0, round(r.left()));   y  = max(0, round(r.top()))
            x2 = min(w, round(r.right()));  y2 = min(h, round(r.bottom()))
            if x2 - x >= MIN_CROP_PX and y2 - y >= MIN_CROP_PX:
                self.grabcut_rect_committed.emit(x, y, x2 - x, y2 - y)
        self._mode = CanvasMode.NORMAL
        self._crop_rect = None; self._active_handle = _Handle.NONE
        self._drag_start_pos = None; self._drag_start_rect = None
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.viewport().update()

    def set_move_mode(self, enabled: bool) -> None:
        self._mode = CanvasMode.MOVE if enabled else CanvasMode.NORMAL
        self.setCursor(Qt.CursorShape.SizeAllCursor if enabled else Qt.CursorShape.ArrowCursor)
        self.viewport().update()

    def set_draw_mode(self, tool: str) -> None:
        """tool = 'brush' | 'eraser' | 'smudge' | 'eyedropper' | 'text' | ''"""
        _map = {
            "brush":      CanvasMode.BRUSH,
            "eraser":     CanvasMode.ERASER,
            "smudge":     CanvasMode.SMUDGE,
            "eyedropper": CanvasMode.EYEDROPPER,
            "text":       CanvasMode.TEXT,
        }
        self._mode = _map.get(tool, CanvasMode.NORMAL)
        self._drawing = False

        cursor_map = {
            CanvasMode.BRUSH:      Qt.CursorShape.BlankCursor,
            CanvasMode.ERASER:     Qt.CursorShape.BlankCursor,
            CanvasMode.SMUDGE:     Qt.CursorShape.BlankCursor,
            CanvasMode.EYEDROPPER: Qt.CursorShape.CrossCursor,
            CanvasMode.TEXT:       Qt.CursorShape.IBeamCursor,
        }
        self.setCursor(cursor_map.get(self._mode, Qt.CursorShape.ArrowCursor))
        self.viewport().update()

    def set_brush_size(self, size: int) -> None:
        self._brush_size = max(1, size)
        self.viewport().update()

    # ── API testo ─────────────────────────────────────────────────────

    def set_text_preview(self, text: str, font_family: str, font_size: int,
                         bold: bool, italic: bool, color: QColor) -> None:
        """Aggiorna i parametri del testo di anteprima senza spostare l'ancora."""
        self._text_preview = dict(text=text, font_family=font_family,
                                  font_size=font_size, bold=bold,
                                  italic=italic, color=QColor(color))
        self.viewport().update()

    def get_text_anchor(self) -> Optional[tuple[float, float]]:
        if self._text_anchor is None: return None
        return (self._text_anchor.x(), self._text_anchor.y())

    def clear_text_anchor(self) -> None:
        self._text_anchor = None
        self.viewport().update()

    # ── API selezione ────────────────────────────────────────────────

    def start_selection_mode(self, shape: str) -> None:
        """shape = 'rect' | 'ellipse' | 'lasso'"""
        _map = {"rect": CanvasMode.SELECT_RECT,
                "ellipse": CanvasMode.SELECT_ELLIPSE,
                "lasso": CanvasMode.SELECT_LASSO}
        self._mode = _map.get(shape, CanvasMode.SELECT_RECT)
        self._sel_start = None
        self._sel_rect = None
        self._sel_lasso_pts = []
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.viewport().update()

    def cancel_selection_mode(self) -> None:
        if self._mode not in (CanvasMode.SELECT_RECT, CanvasMode.SELECT_ELLIPSE,
                               CanvasMode.SELECT_LASSO):
            return
        self._mode = CanvasMode.NORMAL
        self._sel_start = None; self._sel_rect = None; self._sel_lasso_pts = []
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.selection_cancelled.emit()
        self.viewport().update()

    def set_selection_mask(self, mask) -> None:
        """Aggiorna la maschera di selezione per il rendering marching ants."""
        from core.selection import selection_outline_points
        self._sel_mask = mask
        if mask is not None and mask.any():
            self._sel_outline = selection_outline_points(mask)
        else:
            self._sel_outline = []
        self.viewport().update()

    def tick_ants(self) -> None:
        """Chiamato da un QTimer in main_window ogni 80ms per animare i marching ants."""
        if self._sel_outline:
            self._ants_offset = (self._ants_offset + 1) % 8
            self.viewport().update()

    # ── API clone stamp ───────────────────────────────────────────────

    def start_clone_mode(self) -> None:
        self._mode = CanvasMode.CLONE
        self._clone_source = None
        self._clone_drawing = False
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.viewport().update()

    def end_clone_mode(self) -> None:
        self._mode = CanvasMode.NORMAL
        self._clone_source = None
        self._clone_drawing = False
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.viewport().update()

    def set_clone_source(self, x: float, y: float) -> None:
        """Imposta il punto sorgente (chiamato dal main_window quando Alt+click)."""
        self._clone_source = QPointF(x, y)
        self.viewport().update()

    @property
    def mode(self) -> CanvasMode:
        return self._mode

    # ── Events ────────────────────────────────────────────────────────

    def wheelEvent(self, event: QWheelEvent) -> None:
        if self._mode == CanvasMode.CROP: return
        d = event.angleDelta().y()
        nz = self._zoom_factor * ZOOM_STEP if d > 0 else self._zoom_factor / ZOOM_STEP
        nz = max(MIN_ZOOM, min(MAX_ZOOM, nz))
        self.resetTransform(); self.scale(nz, nz)
        self._zoom_factor = nz; self.zoom_changed.emit(nz)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        vp = event.position()
        sp = self.mapToScene(vp.toPoint())

        if self._mode == CanvasMode.MOVE and event.button() == Qt.MouseButton.LeftButton:
            self._move_origin = sp; return

        if self._mode in (CanvasMode.BRUSH, CanvasMode.ERASER, CanvasMode.SMUDGE):
            if event.button() == Qt.MouseButton.LeftButton:
                self._drawing = True
                self.stroke_begin.emit(sp.x(), sp.y())
            return

        if self._mode == CanvasMode.EYEDROPPER and event.button() == Qt.MouseButton.LeftButton:
            if self._image_array is not None:
                x, y = int(sp.x()), int(sp.y())
                h, w = self._image_array.shape[:2]
                if 0 <= x < w and 0 <= y < h:
                    self.eyedropper_pick.emit(x, y)
            return

        if self._mode == CanvasMode.TEXT and event.button() == Qt.MouseButton.LeftButton:
            # Check if clicking near existing anchor → drag
            if self._text_anchor is not None:
                anchor_vp = self.mapFromScene(self._text_anchor)
                dist = ((vp.x() - anchor_vp.x())**2 + (vp.y() - anchor_vp.y())**2) ** 0.5
                if dist <= _TEXT_DRAG_HIT:
                    self._text_dragging = True
                    return
            # Click elsewhere → set new anchor
            self._text_anchor = QPointF(sp)
            self._text_dragging = True
            self.text_anchor_changed.emit(sp.x(), sp.y())
            self.viewport().update()
            return

        if self._mode == CanvasMode.CROP and event.button() == Qt.MouseButton.LeftButton:
            h = self._hit_test(vp)
            if h == _Handle.NONE:
                if self._crop_rect is not None: return   # crop: non si ridisegna
                self._drag_start_pos = sp
                self._active_handle = _Handle.BOTTOM_RIGHT
                self._crop_rect = QRectF(sp, sp)
                self._drag_start_rect = QRectF(self._crop_rect)
            else:
                self._drag_start_pos = sp
                self._active_handle = h
                self._drag_start_rect = QRectF(self._crop_rect)
            return

        if self._mode == CanvasMode.GRABCUT and event.button() == Qt.MouseButton.LeftButton:
            h = self._hit_test(vp)
            if h == _Handle.NONE:
                # GrabCut: nuovo click fuori dai handle → ridisegna da capo
                self._drag_start_pos = sp
                self._active_handle = _Handle.BOTTOM_RIGHT
                self._crop_rect = QRectF(sp, sp)
                self._drag_start_rect = QRectF(self._crop_rect)
            else:
                # Click su un handle: modifica il rettangolo esistente
                self._drag_start_pos = sp
                self._active_handle = h
                self._drag_start_rect = QRectF(self._crop_rect)
            return

        if self._mode in (CanvasMode.SELECT_RECT, CanvasMode.SELECT_ELLIPSE) and                 event.button() == Qt.MouseButton.LeftButton:
            self._sel_start = sp
            self._sel_rect = QRectF(sp, sp)
            return

        if self._mode == CanvasMode.SELECT_LASSO and                 event.button() == Qt.MouseButton.LeftButton:
            self._sel_lasso_pts = [(int(sp.x()), int(sp.y()))]
            return

        if self._mode == CanvasMode.CLONE:
            if event.button() == Qt.MouseButton.LeftButton:
                if event.modifiers() & Qt.KeyboardModifier.AltModifier:
                    # Alt+click imposta la sorgente
                    self.clone_source_set.emit(sp.x(), sp.y())
                    self._clone_source = QPointF(sp)
                    self.viewport().update()
                elif self._clone_source is not None:
                    # Click normale → inizia a dipingere
                    self._clone_drawing = True
                    self.clone_stroke_begin.emit(sp.x(), sp.y())
            return

        if event.button() == Qt.MouseButton.MiddleButton or (
            self._space_held and event.button() == Qt.MouseButton.LeftButton
        ):
            self._panning = True; self._last_pan_pos = vp
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        vp = event.position()
        sp = self.mapToScene(vp.toPoint())

        if self._image_array is not None:
            x, y = int(sp.x()), int(sp.y())
            h, w = self._image_array.shape[:2]
            if 0 <= x < w and 0 <= y < h:
                self.pixel_hovered.emit(x, y)

        if self._mode == CanvasMode.MOVE:
            if self._move_origin is not None:
                dx = int(sp.x() - self._move_origin.x())
                dy = int(sp.y() - self._move_origin.y())
                if dx != 0 or dy != 0:
                    self._move_origin = sp
                    self.layer_moved.emit(dx, dy)
            return

        if self._mode in (CanvasMode.BRUSH, CanvasMode.ERASER, CanvasMode.SMUDGE):
            self._cursor_pos = vp
            if self._drawing:
                self.stroke_move.emit(sp.x(), sp.y())
            self.viewport().update()
            return

        if self._mode == CanvasMode.TEXT:
            self._cursor_pos = vp
            if self._text_dragging and self._text_anchor is not None:
                self._text_anchor = QPointF(sp)
                self.text_anchor_changed.emit(sp.x(), sp.y())
            self.viewport().update()
            return

        if self._mode in (CanvasMode.SELECT_RECT, CanvasMode.SELECT_ELLIPSE):
            if self._sel_start is not None:
                self._sel_rect = QRectF(self._sel_start, sp).normalized()
                self.viewport().update()
            return

        if self._mode == CanvasMode.SELECT_LASSO:
            if self._sel_lasso_pts:
                self._sel_lasso_pts.append((int(sp.x()), int(sp.y())))
                self.viewport().update()
            return

        if self._mode == CanvasMode.CLONE:
            self._cursor_pos = vp
            if self._clone_drawing:
                self.clone_stroke_move.emit(sp.x(), sp.y())
            self.viewport().update()
            return

        if self._mode in (CanvasMode.CROP, CanvasMode.GRABCUT):
            if self._drag_start_pos is not None and self._drag_start_rect is not None:
                self._apply_drag(sp); self.viewport().update()
            else:
                if self._mode == CanvasMode.GRABCUT:
                    self.setCursor(Qt.CursorShape.CrossCursor)
                else:
                    self.setCursor(_HANDLE_CURSORS[self._hit_test(vp)])
            return

        if self._panning and self._last_pan_pos is not None:
            delta = vp - self._last_pan_pos; self._last_pan_pos = vp
            self.horizontalScrollBar().setValue(int(self.horizontalScrollBar().value() - delta.x()))
            self.verticalScrollBar().setValue(int(self.verticalScrollBar().value() - delta.y()))
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._mode == CanvasMode.MOVE:
            if event.button() == Qt.MouseButton.LeftButton: self._move_origin = None
            return

        if self._mode in (CanvasMode.BRUSH, CanvasMode.ERASER, CanvasMode.SMUDGE):
            if event.button() == Qt.MouseButton.LeftButton and self._drawing:
                self._drawing = False; self.stroke_end.emit()
            return

        if self._mode == CanvasMode.TEXT:
            if event.button() == Qt.MouseButton.LeftButton:
                self._text_dragging = False
            return

        if self._mode in (CanvasMode.SELECT_RECT, CanvasMode.SELECT_ELLIPSE):
            if event.button() == Qt.MouseButton.LeftButton and self._sel_rect is not None:
                r = self._sel_rect.normalized()
                if self._image_array is not None:
                    h, w = self._image_array.shape[:2]
                    x  = max(0, int(r.left()));  y  = max(0, int(r.top()))
                    x2 = min(w, int(r.right())); y2 = min(h, int(r.bottom()))
                    if x2 > x and y2 > y:
                        if self._mode == CanvasMode.SELECT_RECT:
                            self.selection_rect_ready.emit(x, y, x2 - x, y2 - y)
                        else:
                            self.selection_ellipse_ready.emit(x, y, x2 - x, y2 - y)
                self._sel_start = None
                self._sel_rect = None
            return

        if self._mode == CanvasMode.SELECT_LASSO:
            if event.button() == Qt.MouseButton.LeftButton and len(self._sel_lasso_pts) >= 3:
                self.selection_lasso_ready.emit(list(self._sel_lasso_pts))
            self._sel_lasso_pts = []
            return

        if self._mode == CanvasMode.CLONE:
            if event.button() == Qt.MouseButton.LeftButton and self._clone_drawing:
                self._clone_drawing = False
                self.clone_stroke_end.emit()
            return

        if self._mode in (CanvasMode.CROP, CanvasMode.GRABCUT):
            if event.button() == Qt.MouseButton.LeftButton:
                self._active_handle = _Handle.NONE
                self._drag_start_pos = None; self._drag_start_rect = None
                if self._mode == CanvasMode.GRABCUT:
                    self.setCursor(Qt.CursorShape.CrossCursor)
                else:
                    self.setCursor(_HANDLE_CURSORS[self._hit_test(event.position())])
            return

        if self._panning:
            self._panning = False; self._last_pan_pos = None
            self.setCursor(Qt.CursorShape.OpenHandCursor if self._space_held
                           else Qt.CursorShape.ArrowCursor)
        else:
            super().mouseReleaseEvent(event)

    def leaveEvent(self, event) -> None:
        self._cursor_pos = None
        self.viewport().update()
        super().leaveEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if self._mode == CanvasMode.CROP:
            if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                self.end_crop_mode(confirm=True)
            elif event.key() == Qt.Key.Key_Escape:
                self.end_crop_mode(confirm=False)
            return
        if self._mode == CanvasMode.GRABCUT:
            if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                self.end_grabcut_mode(confirm=True)
            elif event.key() == Qt.Key.Key_Escape:
                self.end_grabcut_mode(confirm=False)
            return
        if self._mode in (CanvasMode.SELECT_RECT, CanvasMode.SELECT_ELLIPSE,
                           CanvasMode.SELECT_LASSO):
            if event.key() == Qt.Key.Key_Escape:
                self.cancel_selection_mode()
            return
        if event.key() == Qt.Key.Key_M and not event.isAutoRepeat():
            self.set_move_mode(self._mode != CanvasMode.MOVE); return
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space_held = True
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space_held = False
            if not self._panning: self.setCursor(Qt.CursorShape.ArrowCursor)
        else:
            super().keyReleaseEvent(event)

    # ── Overlay (drawForeground) ──────────────────────────────────────

    def drawForeground(self, painter: QPainter, scene_rect: QRectF) -> None:
        super().drawForeground(painter, scene_rect)

        # Gli overlay in coordinate viewport usano resetTransform() internamente.
        # Il crop overlay usa le coordinate scena native del painter.

        if self._mode == CanvasMode.MOVE:
            self._draw_badge(painter, "✥ Move  (M per uscire)", QColor(80, 160, 255))

        if self._mode in (CanvasMode.BRUSH, CanvasMode.ERASER, CanvasMode.SMUDGE):
            self._draw_brush_cursor(painter)
            labels = {"brush": "🖌 Pennello", "eraser": "⬜ Gomma", "smudge": "💧 Sfumino"}
            label = labels.get(
                "brush" if self._mode == CanvasMode.BRUSH else
                "eraser" if self._mode == CanvasMode.ERASER else "smudge", "")
            self._draw_badge(painter, label, QColor(255, 180, 60))

        if self._mode == CanvasMode.EYEDROPPER:
            self._draw_badge(painter, "🔍 Contagocce — clicca per prelevare il colore",
                             QColor(100, 220, 100))

        if self._mode == CanvasMode.GRABCUT:
            self._draw_badge(painter, "✂ GrabCut — disegna un rettangolo (Invio=applica, Esc=annulla)",
                             QColor(60, 220, 100))

        if self._mode == CanvasMode.TEXT:
            self._draw_text_overlay(painter)
            self._draw_badge(painter, "T Testo — clicca per posizionare", QColor(160, 120, 255))

        # Selezione overlay (coordinate SCENA per il rettangolo/lasso)
        if self._mode in (CanvasMode.SELECT_RECT, CanvasMode.SELECT_ELLIPSE):
            self._draw_badge(painter, "Seleziona — trascina (Esc=annulla)", QColor(255, 220, 40))
            if self._sel_rect is not None:
                self._draw_selection_rect(painter, self._sel_rect,
                                          ellipse=(self._mode == CanvasMode.SELECT_ELLIPSE))
        if self._mode == CanvasMode.SELECT_LASSO:
            self._draw_badge(painter, "Lasso — trascina (Esc=annulla)", QColor(255, 220, 40))
            if len(self._sel_lasso_pts) >= 2:
                self._draw_lasso(painter, self._sel_lasso_pts)

        # Marching ants selezione attiva
        if self._sel_outline:
            self._draw_marching_ants(painter)

        # Clone stamp overlay
        if self._mode == CanvasMode.CLONE:
            self._draw_clone_overlay(painter)

        # Crop / GrabCut overlay (coordinate SCENA)
        if self._mode == CanvasMode.CROP and self._crop_rect is not None:
            self._draw_crop_overlay(painter, scene_rect)
        if self._mode == CanvasMode.GRABCUT and self._crop_rect is not None:
            self._draw_grabcut_overlay(painter, scene_rect)

    def _draw_brush_cursor(self, painter: QPainter) -> None:
        """Cursore pennello in coordinate VIEWPORT (resetTransform)."""
        if self._cursor_pos is None: return
        painter.save()
        painter.resetTransform()   # coordinate schermo
        r = max(1.0, (self._brush_size / 2.0) * self._zoom_factor)
        cx, cy = self._cursor_pos.x(), self._cursor_pos.y()
        is_eraser = (self._mode == CanvasMode.ERASER)
        painter.setPen(QPen(QColor(0, 0, 0, 180), 1.5))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(QPointF(cx, cy), r + 1, r + 1)
        inner = QColor(255, 80, 80, 220) if is_eraser else QColor(255, 255, 255, 220)
        painter.setPen(QPen(inner, 1.0))
        painter.drawEllipse(QPointF(cx, cy), r, r)
        ch = 4
        painter.drawLine(QPointF(cx - ch, cy), QPointF(cx + ch, cy))
        painter.drawLine(QPointF(cx, cy - ch), QPointF(cx, cy + ch))
        painter.restore()

    def _draw_text_overlay(self, painter: QPainter) -> None:
        """Anteprima del testo nel punto di ancoraggio (coordinate VIEWPORT)."""
        painter.save()
        painter.resetTransform()   # coordinate schermo

        if self._text_anchor is None:
            # Mostra solo cursore a croce dove si trova il mouse
            if self._cursor_pos is not None:
                painter.setPen(QPen(QColor(160, 120, 255, 160), 1.5))
                sz = 10
                cx, cy = self._cursor_pos.x(), self._cursor_pos.y()
                painter.drawLine(QPointF(cx - sz, cy), QPointF(cx + sz, cy))
                painter.drawLine(QPointF(cx, cy - sz), QPointF(cx, cy + sz))
            painter.restore()
            return

        # Converti ancora da scene → viewport (mapFromScene è affidabile)
        anchor_vp = self.mapFromScene(self._text_anchor)
        ax, ay = anchor_vp.x(), anchor_vp.y()

        tp = self._text_preview
        text = tp.get("text", "")
        font_family = tp.get("font_family", "Arial")
        font_size = tp.get("font_size", 24)
        bold = tp.get("bold", False)
        italic = tp.get("italic", False)
        color: QColor = tp.get("color", QColor(0, 0, 0, 255))

        font = QFont(font_family)
        font.setPixelSize(max(4, int(font_size * self._zoom_factor)))
        font.setBold(bold)
        font.setItalic(italic)
        painter.setFont(font)

        if text:
            # Anteprima testo
            painter.setPen(QPen(color))
            painter.drawText(QPointF(ax, ay), text)

            # Rettangolo tratteggiato intorno al testo
            fm = painter.fontMetrics()
            tw = fm.horizontalAdvance(text)
            th = fm.height()
            margin = 4
            rect = QRectF(ax - margin, ay - fm.ascent() - margin,
                          tw + margin * 2, th + margin * 2)
            pen = QPen(QColor(160, 120, 255, 200), 1.5)
            pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(rect)
        else:
            # Placeholder tratteggiato
            painter.setPen(QPen(QColor(160, 120, 255, 120), 1.5, Qt.PenStyle.DashLine))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(QRectF(ax - 4, ay - 20, 120, 30))
            painter.setPen(QColor(160, 120, 255, 100))
            painter.drawText(QPointF(ax, ay), "Inserisci testo nel pannello →")

        # Maniglietta di ancoraggio
        painter.setPen(QPen(QColor(160, 120, 255, 220), 1.5))
        painter.setBrush(QBrush(QColor(255, 255, 255, 200)))
        painter.drawEllipse(QPointF(ax, ay), 5, 5)
        painter.restore()

    def _draw_badge(self, painter: QPainter, text: str, color: QColor) -> None:
        """Bordo colorato + etichetta in alto a sinistra — coordinate VIEWPORT."""
        painter.save()
        painter.resetTransform()   # coordinate schermo
        vp = self.viewport().rect()
        painter.setPen(QPen(color, 2))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(vp.adjusted(2, 2, -2, -2))
        painter.setPen(color)
        painter.drawText(8, 20, text)
        painter.restore()

    def _draw_selection_rect(self, painter: QPainter, rect: QRectF, ellipse: bool = False) -> None:
        """Rettangolo/ellisse di selezione in costruzione — coordinate scena."""
        zoom = max(self._zoom_factor, 1e-4)
        pen = QPen(QColor(255, 220, 40, 220), 1.5 / zoom)
        pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.setBrush(QBrush(QColor(255, 220, 40, 30)))
        if ellipse:
            painter.drawEllipse(rect)
        else:
            painter.drawRect(rect)

    def _draw_lasso(self, painter: QPainter, pts: list) -> None:
        """Poligono lasso in costruzione — coordinate scena."""
        if len(pts) < 2: return
        zoom = max(self._zoom_factor, 1e-4)
        pen = QPen(QColor(255, 220, 40, 200), 1.5 / zoom)
        pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        from PyQt6.QtGui import QPolygonF
        poly = QPolygonF([QPointF(float(x), float(y)) for x, y in pts])
        painter.drawPolyline(poly)
        # Linea di chiusura tratteggiata
        if len(pts) >= 3:
            pen2 = QPen(QColor(255, 220, 40, 100), 1.0 / zoom)
            pen2.setStyle(Qt.PenStyle.DotLine)
            painter.setPen(pen2)
            painter.drawLine(QPointF(float(pts[-1][0]), float(pts[-1][1])),
                             QPointF(float(pts[0][0]), float(pts[0][1])))

    def _draw_marching_ants(self, painter: QPainter) -> None:
        """Bordo animato della selezione attiva — coordinate scena."""
        if not self._sel_outline: return
        zoom = max(self._zoom_factor, 1e-4)
        off = self._ants_offset

        # Linea bianca solida sotto
        painter.setPen(QPen(QColor(255, 255, 255, 200), 1.0 / zoom))
        for x1, y1, x2, y2 in self._sel_outline:
            painter.drawLine(QPointF(float(x1), float(y1)), QPointF(float(x2), float(y2)))

        # Linea nera tratteggiata sopra (animata)
        pen = QPen(QColor(0, 0, 0, 200), 1.0 / zoom)
        pen.setStyle(Qt.PenStyle.CustomDashLine)
        pen.setDashPattern([4, 4])
        pen.setDashOffset(off)
        painter.setPen(pen)
        for x1, y1, x2, y2 in self._sel_outline:
            painter.drawLine(QPointF(float(x1), float(y1)), QPointF(float(x2), float(y2)))

    def _draw_clone_overlay(self, painter: QPainter) -> None:
        """Badge + cursore pennello + mirino sorgente per clone stamp."""
        if self._clone_source is None:
            self._draw_badge(painter, "Clone — Alt+click per impostare la sorgente",
                             QColor(180, 120, 255))
        else:
            self._draw_badge(painter, "Clone — dipingi (Alt+click per cambiare sorgente)",
                             QColor(180, 120, 255))
            # Mirino sorgente in coordinate viewport
            painter.save()
            painter.resetTransform()
            src_vp = self.mapFromScene(self._clone_source)
            sx, sy = src_vp.x(), src_vp.y()
            r = max(1.0, (self._brush_size / 2.0) * self._zoom_factor)
            painter.setPen(QPen(QColor(0, 0, 0, 180), 1.5))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(QPointF(sx, sy), r + 1, r + 1)
            painter.setPen(QPen(QColor(180, 120, 255, 220), 1.0))
            painter.drawEllipse(QPointF(sx, sy), r, r)
            ch = 6
            painter.drawLine(QPointF(sx - ch, sy), QPointF(sx + ch, sy))
            painter.drawLine(QPointF(sx, sy - ch), QPointF(sx, sy + ch))
            painter.restore()
        # Cursore destinazione (pennello standard)
        self._draw_brush_cursor(painter)

    def _draw_grabcut_overlay(self, painter: QPainter, scene_rect: QRectF) -> None:
        """Overlay GrabCut: rettangolo verde tratteggiato, sfondo scuro."""
        if self._crop_rect is None: return
        zoom = max(self._zoom_factor, 1e-4)
        cr = self._crop_rect.normalized()
        painter.setBrush(QBrush(QColor(0, 0, 0, 140)))
        painter.setPen(Qt.PenStyle.NoPen)
        _fill_outside(painter, scene_rect, cr)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        pen = QPen(QColor(60, 220, 100, 230), 2.0 / zoom)
        pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.drawRect(cr)
        # Angoli solidi
        painter.setPen(QPen(QColor(60, 220, 100, 230), 2.0 / zoom))
        corner_len = min(cr.width(), cr.height()) * 0.12
        for cx, cy in [(cr.left(), cr.top()), (cr.right(), cr.top()),
                       (cr.left(), cr.bottom()), (cr.right(), cr.bottom())]:
            sx = 1 if cx == cr.left() else -1
            sy = 1 if cy == cr.top() else -1
            painter.drawLine(QPointF(cx, cy), QPointF(cx + sx * corner_len, cy))
            painter.drawLine(QPointF(cx, cy), QPointF(cx, cy + sy * corner_len))

    def _draw_crop_overlay(self, painter: QPainter, scene_rect: QRectF) -> None:
        zoom = max(self._zoom_factor, 1e-4)
        cr = self._crop_rect.normalized()
        painter.setBrush(QBrush(QColor(0, 0, 0, 160)))
        painter.setPen(Qt.PenStyle.NoPen)
        _fill_outside(painter, scene_rect, cr)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor(255, 255, 255, 220), 1.5 / zoom))
        painter.drawRect(cr)
        thin = QPen(QColor(255, 255, 255, 70), 0.8 / zoom)
        painter.setPen(thin)
        for i in (1, 2):
            tx = cr.left() + cr.width()  * i / 3
            ty = cr.top()  + cr.height() * i / 3
            painter.drawLine(QPointF(tx, cr.top()),  QPointF(tx, cr.bottom()))
            painter.drawLine(QPointF(cr.left(), ty), QPointF(cr.right(), ty))
        hr = HANDLE_RADIUS / zoom
        painter.setPen(QPen(QColor(255, 255, 255, 230), 1.5 / zoom))
        painter.setBrush(QBrush(QColor(255, 255, 255, 220)))
        for pt in self._handle_points(cr).values():
            painter.drawRect(QRectF(pt.x() - hr, pt.y() - hr, hr * 2, hr * 2))

    # ── Helpers crop ──────────────────────────────────────────────────

    def _handle_points(self, cr: QRectF) -> dict[_Handle, QPointF]:
        cx, cy = cr.center().x(), cr.center().y()
        return {
            _Handle.TOP_LEFT:     QPointF(cr.left(),  cr.top()),
            _Handle.TOP:          QPointF(cx,          cr.top()),
            _Handle.TOP_RIGHT:    QPointF(cr.right(), cr.top()),
            _Handle.RIGHT:        QPointF(cr.right(), cy),
            _Handle.BOTTOM_RIGHT: QPointF(cr.right(), cr.bottom()),
            _Handle.BOTTOM:       QPointF(cx,          cr.bottom()),
            _Handle.BOTTOM_LEFT:  QPointF(cr.left(),  cr.bottom()),
            _Handle.LEFT:         QPointF(cr.left(),  cy),
        }

    def _hit_test(self, vp_pos: QPointF) -> _Handle:
        if self._crop_rect is None: return _Handle.NONE
        sp = self.mapToScene(vp_pos.toPoint())
        cr = self._crop_rect.normalized()
        hit = HANDLE_HIT / max(self._zoom_factor, 1e-4)
        for h, pt in self._handle_points(cr).items():
            if abs(sp.x() - pt.x()) <= hit and abs(sp.y() - pt.y()) <= hit:
                return h
        if cr.contains(sp): return _Handle.MOVE
        return _Handle.NONE

    def _apply_drag(self, sp: QPointF) -> None:
        if (self._drag_start_pos is None or self._drag_start_rect is None
                or self._image_array is None):
            return
        h, w = self._image_array.shape[:2]
        dx = sp.x() - self._drag_start_pos.x()
        dy = sp.y() - self._drag_start_pos.y()
        r = QRectF(self._drag_start_rect)
        hd = self._active_handle
        if hd == _Handle.MOVE:
            r.translate(dx, dy)
            if r.left() < 0: r.translate(-r.left(), 0)
            if r.top()  < 0: r.translate(0, -r.top())
            if r.right()  > w: r.translate(w - r.right(), 0)
            if r.bottom() > h: r.translate(0, h - r.bottom())
        else:
            l, t, rr, b = r.left(), r.top(), r.right(), r.bottom()
            if hd in (_Handle.LEFT, _Handle.TOP_LEFT, _Handle.BOTTOM_LEFT):
                l = max(0.0, min(l + dx, rr - MIN_CROP_PX))
            if hd in (_Handle.RIGHT, _Handle.TOP_RIGHT, _Handle.BOTTOM_RIGHT):
                rr = min(float(w), max(rr + dx, l + MIN_CROP_PX))
            if hd in (_Handle.TOP, _Handle.TOP_LEFT, _Handle.TOP_RIGHT):
                t = max(0.0, min(t + dy, b - MIN_CROP_PX))
            if hd in (_Handle.BOTTOM, _Handle.BOTTOM_LEFT, _Handle.BOTTOM_RIGHT):
                b = min(float(h), max(b + dy, t + MIN_CROP_PX))
            r = QRectF(QPointF(l, t), QPointF(rr, b))
        self._crop_rect = r

    @staticmethod
    def _to_pixmap(array: np.ndarray) -> QPixmap:
        h, w, ch = array.shape; assert ch == 4
        return QPixmap.fromImage(QImage(array.tobytes(), w, h, w * 4,
                                        QImage.Format.Format_RGBA8888))


def _fill_outside(painter: QPainter, outer: QRectF, inner: QRectF) -> None:
    if inner.top()    > outer.top():
        painter.drawRect(QRectF(outer.left(), outer.top(), outer.width(), inner.top() - outer.top()))
    if inner.bottom() < outer.bottom():
        painter.drawRect(QRectF(outer.left(), inner.bottom(), outer.width(), outer.bottom() - inner.bottom()))
    if inner.left()   > outer.left():
        painter.drawRect(QRectF(outer.left(), inner.top(), inner.left() - outer.left(), inner.height()))
    if inner.right()  < outer.right():
        painter.drawRect(QRectF(inner.right(), inner.top(), outer.right() - inner.right(), inner.height()))
