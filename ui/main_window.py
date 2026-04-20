from __future__ import annotations

from pathlib import Path as FilePath
from typing import Optional
import numpy as np
from PyQt6.QtCore import Qt, QSize, QThread, QObject, pyqtSignal
from PyQt6.QtGui import QAction, QKeySequence, QColor, QImage
from PyQt6.QtWidgets import (
    QMainWindow, QFileDialog, QMessageBox, QLabel, QInputDialog,
    QToolBar, QStatusBar, QSizePolicy, QDockWidget, QTabWidget,
    QApplication,
)

from core.history import HistoryManager
from core.filter_stack import FilterStack
from core.layer_stack import LayerStack, Path as LayerPath
from core.layer import Layer, LayerGroup
from core.drawing import paint_stamp, interpolate_stroke
from core.drawing_tools import apply_smudge, apply_gradient, build_heal_mask, apply_healing
from core.drawing_engine import draw_text
from core.selection import (mask_from_rect, mask_from_ellipse, mask_from_lasso,
                            invert_mask, select_all, copy_region, cut_region,
                            fill_region, feather_mask)
from core.clone_stamp import clone_stamp_dab, interpolate_clone_stroke
from ui.canvas import ImageCanvas, CanvasMode
from ui.filter_panel import FilterPanel
from ui.layer_panel import LayerPanel
from ui.transform_panel import TransformPanel
from ui.export_panel import ExportPanel, FORMATS
from ui.drawing_panel import DrawingPanel
from utils.file_io import load_image, save_image
from utils.project_io import save_project, load_project
from utils.ora_io import save_ora, load_ora
from ui.matting_panel import MattingPanel
from ui.adjustment_panel import AdjustmentPanel
from ui.free_transform_panel import FreeTransformPanel
from core.layer import AdjustmentLayer
from core.free_transform import apply_transform
from ui.history_panel import HistoryPanel
from ui.upscale_panel import UpscalePanel, _UpscaleWorker
from core.magic_wand import magic_wand_select
from core.raw_io import is_raw, load_raw, RAW_EXTENSIONS
from core.shapes import (Shape, ShapeKind, ShapeStyle, HandleID,
                          default_shape, render_shape_onto)


class _RembgWorker(QObject):
    """Worker che esegue rembg in un QThread separato."""
    done  = pyqtSignal(object)   # np.ndarray risultante
    error = pyqtSignal(str)

    def __init__(self, arr: np.ndarray, model: str, alpha_matting: bool,
                 fg_thresh: int, bg_thresh: int, erode: int) -> None:
        super().__init__()
        self._arr  = arr
        self._model = model
        self._am   = alpha_matting
        self._fg   = fg_thresh
        self._bg   = bg_thresh
        self._er   = erode

    def run(self) -> None:
        try:
            from core.matting import remove_background_rembg
            result = remove_background_rembg(
                self._arr, self._model, self._am, self._fg, self._bg, self._er)
            self.done.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))


class _MagicWandWorker(QObject):
    """Esegue magic_wand_select in un QThread separato per non bloccare la UI."""
    done  = pyqtSignal(object)   # np.ndarray bool (H, W)
    error = pyqtSignal(str)

    def __init__(self,
                 rgba: np.ndarray,
                 lx: int, ly: int,
                 offset_x: int, offset_y: int,
                 canvas_w: int, canvas_h: int,
                 tolerance: int, contiguous: bool, anti_alias: bool) -> None:
        super().__init__()
        self._rgba      = rgba
        self._lx        = lx
        self._ly        = ly
        self._ox        = offset_x
        self._oy        = offset_y
        self._cw        = canvas_w
        self._ch        = canvas_h
        self._tolerance = tolerance
        self._contiguous = contiguous
        self._anti_alias = anti_alias

    def run(self) -> None:
        try:
            mask_layer = magic_wand_select(
                self._rgba, self._lx, self._ly,
                tolerance=self._tolerance,
                contiguous=self._contiguous,
                anti_alias=self._anti_alias,
            )
            lh, lw = self._rgba.shape[:2]
            mask_canvas = np.zeros((self._ch, self._cw), dtype=bool)
            cy0 = max(0, self._oy)
            cx0 = max(0, self._ox)
            cy1 = min(self._ch, self._oy + lh)
            cx1 = min(self._cw, self._ox + lw)
            ly0 = cy0 - self._oy
            lx0 = cx0 - self._ox
            mask_canvas[cy0:cy1, cx0:cx1] = mask_layer[
                ly0:ly0 + (cy1 - cy0), lx0:lx0 + (cx1 - cx0)]
            self.done.emit(mask_canvas)
        except Exception as exc:
            self.error.emit(str(exc))


def _union_dirty(dirty_iter) -> tuple | None:
    """Unisce più DirtyRect (lx0,ly0,lx1,ly1) in un unico bounding box."""
    x0 = y0 = float('inf')
    x1 = y1 = float('-inf')
    found = False
    for d in dirty_iter:
        if not d or d == (0, 0, 0, 0):
            continue
        found = True
        x0 = min(x0, d[0]); y0 = min(y0, d[1])
        x1 = max(x1, d[2]); y1 = max(y1, d[3])
    return (int(x0), int(y0), int(x1), int(y1)) if found else None


_SHAPE_KEY_MAP: dict[str, ShapeKind] = {
    "ellipse":  ShapeKind.ELLIPSE,
    "rect":     ShapeKind.RECT,
    "polygon":  ShapeKind.POLYGON,
    "star":     ShapeKind.STAR,
    "squircle": ShapeKind.SQUIRCLE,
    "arrow":    ShapeKind.ARROW,
    "svg":      ShapeKind.SVG,
}


def _shape_key_to_kind(key: str) -> ShapeKind:
    return _SHAPE_KEY_MAP.get(key, ShapeKind.ELLIPSE)


def _apply_panel_to_shape(dp, shape: Shape) -> None:
    """
    Copia tutti i parametri dalla DrawingPanel sulla shape.
    Separato da MainWindow per leggibilità e testabilità.
    """
    from core.shapes import ShapeStyle
    shape.style = ShapeStyle(
        fill_color   = dp.shape_fill_color,
        stroke_color = dp.shape_stroke_color,
        stroke_width = dp.shape_stroke_width,
        opacity      = dp.shape_opacity,
    )
    shape.n_sides      = dp.shape_n_sides
    shape.inner_ratio  = dp.shape_inner_ratio
    shape.sq_exp       = dp.shape_sq_exp
    shape.corner_r     = dp.shape_corner_r
    shape.arrow_width  = dp.shape_arrow_width
    # SVG: percorso file e flag tint
    if shape.kind == ShapeKind.SVG:
        svg_path = dp.get_svg_path()
        if svg_path:
            shape.svg_path  = svg_path
        shape.svg_tint  = dp.shape_svg_tint


class MainWindow(QMainWindow):

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Image Suite")
        self.resize(1440, 900)

        self._current_path: FilePath | None = None
        self._project_path: FilePath | None = None   # path del file .isp
        self._history = HistoryManager(max_steps=30)
        self._unsaved_changes = False
        self._filters = FilterStack()
        self._preview: np.ndarray | None = None
        self._layer_stack: LayerStack | None = None
        self._last_stroke_pt: Optional[tuple[float, float]] = None
        self._prev_draw_tool: str = ""

        # Selezione
        self._sel_mask: Optional[np.ndarray] = None     # bool (H,W) o None
        self._sel_clipboard: Optional[np.ndarray] = None  # RGBA patch copiato
        self._sel_feather: int = 0

        # Free transform
        self._ft_original_data: Optional[np.ndarray] = None   # dati pre-transform

        # Clone stamp
        self._clone_source_canvas: Optional[tuple[float,float]] = None  # canvas coords
        self._clone_source_layer: Optional[tuple[float,float]] = None   # layer coords
        self._clone_last_dst: Optional[tuple[float,float]] = None
        self._clone_last_src: Optional[tuple[float,float]] = None
        self._clone_timer_id: int = 0
        self._draw_bg_cache: np.ndarray | None = None
        self._canvas_img_buf: np.ndarray | None = None
        self._rembg_thread: QThread | None = None
        self._rembg_worker: _RembgWorker | None = None
        self._heal_mask_acc: np.ndarray | None = None   # maschera cumulativa del tratto heal
        self._wand_thread:  QThread | None = None
        self._wand_worker:  _MagicWandWorker | None = None
        self._current_shape: Shape | None = None
        self._svg_path:      str = ""

        self._canvas = ImageCanvas()
        self.setCentralWidget(self._canvas)

        self._filter_panel    = FilterPanel()
        self._transform_panel = TransformPanel()
        self._export_panel    = ExportPanel()
        self._layer_panel     = LayerPanel()
        self._drawing_panel   = DrawingPanel()
        self._matting_panel      = MattingPanel()
        self._adjustment_panel   = AdjustmentPanel()
        self._free_transform_panel = FreeTransformPanel()
        self._upscale_panel      = UpscalePanel()

        tabs = QTabWidget()
        tabs.setMinimumWidth(290)
        tabs.addTab(self._drawing_panel,        "✏ Disegno")
        tabs.addTab(self._matting_panel,        "✂ Scontorna")
        tabs.addTab(self._adjustment_panel,     "⚙ Regolazioni")
        tabs.addTab(self._free_transform_panel, "⇲ Trasforma")
        tabs.addTab(self._upscale_panel,        "🔍 Upscale")
        tabs.addTab(self._layer_panel,          "🗂 Layer")
        tabs.addTab(self._filter_panel,         "🎨 Filtri")
        tabs.addTab(self._transform_panel,      "✂ Ritocchi")
        tabs.addTab(self._export_panel,         "💾 Esporta")

        self._dock = QDockWidget("Strumenti", self)
        self._dock.setWidget(tabs)
        self._dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._dock)

        # ── Storico visuale (dock separato, nascosto di default) ──────
        self._history_panel = HistoryPanel()
        self._history_dock = QDockWidget("Storico", self)
        self._history_dock.setWidget(self._history_panel)
        self._history_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea |
            Qt.DockWidgetArea.BottomDockWidgetArea)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self._history_dock)
        self._history_dock.setVisible(False)
        self._history_panel.jump_requested.connect(self._on_history_jump)
        self._history_dock.visibilityChanged.connect(
            lambda v: self._refresh_history_panel() if v else None)

        # ── Segnali canvas ────────────────────────────────────────────
        self._canvas.zoom_changed.connect(self._on_zoom_changed)
        self._canvas.pixel_hovered.connect(self._on_pixel_hovered)
        self._canvas.canvas_cleared.connect(self._on_canvas_cleared)
        self._canvas.crop_committed.connect(self._on_crop_committed)
        self._canvas.layer_moved.connect(self._on_layer_moved)
        self._canvas.stroke_begin.connect(self._on_stroke_begin)
        self._canvas.stroke_move.connect(self._on_stroke_move)
        self._canvas.stroke_end.connect(self._on_stroke_end)
        self._canvas.selection_rect_ready.connect(self._on_selection_rect)
        self._canvas.selection_ellipse_ready.connect(self._on_selection_ellipse)
        self._canvas.selection_lasso_ready.connect(self._on_selection_lasso)
        self._canvas.selection_cancelled.connect(self._on_selection_cancelled)
        self._canvas.clone_source_set.connect(self._on_clone_source_set)
        self._canvas.clone_stroke_begin.connect(self._on_clone_begin)
        self._canvas.clone_stroke_move.connect(self._on_clone_move)
        self._canvas.clone_stroke_end.connect(self._on_clone_end)
        self._canvas.eyedropper_pick.connect(self._on_eyedropper_pick)
        self._canvas.text_anchor_changed.connect(self._on_text_anchor_changed)
        self._canvas.gradient_applied.connect(self._on_gradient_applied)
        self._canvas.heal_stroke_begin.connect(self._on_heal_begin)
        self._canvas.heal_stroke_move.connect(self._on_heal_move)
        self._canvas.heal_stroke_end.connect(self._on_heal_end)
        self._canvas.file_dropped.connect(self._on_file_dropped)
        self._canvas.image_dropped.connect(self._on_image_dropped)
        self._canvas.magic_wand_clicked.connect(self._on_magic_wand_click)
        self._upscale_panel.upscale_requested.connect(self._on_upscale_requested)

        # Stato upscale thread
        self._upscale_thread: QThread | None = None
        self._upscale_worker: _UpscaleWorker | None = None

        # ── Segnali Forme (da DrawingPanel) ──────────────────────────
        self._drawing_panel.shape_changed.connect(self._on_shape_params_changed)
        self._drawing_panel.shape_apply.connect(self._on_shape_apply)
        self._drawing_panel.shape_kind_changed.connect(self._on_shape_kind_changed)
        self._canvas.shape_placed.connect(self._on_shape_placed)
        self._canvas.shape_handle_drag.connect(self._on_shape_handle_drag)

        # ── Segnali pannello disegno ──────────────────────────────────
        self._drawing_panel.tool_changed.connect(self._on_draw_tool_changed)
        self._drawing_panel.color_changed.connect(self._on_draw_color_changed)
        self._drawing_panel.size_changed.connect(self._on_draw_size_changed)
        self._drawing_panel.hardness_changed.connect(self._on_draw_hardness_changed)
        self._drawing_panel.opacity_changed.connect(self._on_draw_opacity_changed)
        self._drawing_panel.text_apply.connect(self._apply_text)
        self._drawing_panel.text_params_changed.connect(self._on_text_params_changed)
        self._drawing_panel.sel_copy_requested.connect(self._on_sel_copy)
        self._drawing_panel.sel_cut_requested.connect(self._on_sel_cut)
        self._drawing_panel.sel_fill_requested.connect(self._on_sel_fill)
        self._drawing_panel.sel_invert_requested.connect(self._on_sel_invert)
        self._drawing_panel.sel_deselect_requested.connect(self._on_sel_deselect)
        self._drawing_panel.sel_feather_changed.connect(self._on_sel_feather_changed)

        # ── Segnali MattingPanel ────────────────────────────────────────
        self._matting_panel.rembg_requested.connect(self._on_rembg_requested)
        self._matting_panel.grabcut_start.connect(self._on_grabcut_start)
        self._matting_panel.grabcut_cancel.connect(self._on_grabcut_cancel)
        self._matting_panel.grabcut_apply.connect(self._on_grabcut_apply)
        self._matting_panel.refine_requested.connect(self._on_refine_mask)
        self._canvas.grabcut_rect_committed.connect(self._on_grabcut_rect)

        # ── Segnali layer panel ───────────────────────────────────────
        self._layer_panel.active_changed.connect(self._on_layer_active_changed)
        self._layer_panel.visibility_toggled.connect(self._on_layer_visibility_toggled)
        self._layer_panel.opacity_changed.connect(self._on_layer_opacity_changed)
        self._layer_panel.blend_changed.connect(self._on_layer_blend_changed)
        self._layer_panel.add_requested.connect(self._on_layer_add)
        self._layer_panel.delete_requested.connect(self._on_layer_delete)
        self._layer_panel.duplicate_requested.connect(self._on_layer_duplicate)
        self._layer_panel.move_up_requested.connect(self._on_layer_move_up)
        self._layer_panel.move_down_requested.connect(self._on_layer_move_down)
        self._layer_panel.rename_requested.connect(self._on_layer_rename)
        self._layer_panel.import_as_layer.connect(self._on_import_as_layer)
        self._layer_panel.create_group_requested.connect(self._on_create_group)
        self._layer_panel.group_node_requested.connect(self._on_group_node)
        self._layer_panel.ungroup_requested.connect(self._on_ungroup)
        self._layer_panel.move_into_group_req.connect(self._on_move_into_group)
        self._layer_panel.move_out_of_group_req.connect(self._on_move_out_of_group)
        self._layer_panel.mask_add_white.connect(lambda p: self._on_mask_op(p, "white"))
        self._layer_panel.mask_add_black.connect(lambda p: self._on_mask_op(p, "black"))
        self._layer_panel.mask_invert.connect(lambda p: self._on_mask_op(p, "invert"))
        self._layer_panel.mask_apply.connect(lambda p: self._on_mask_op(p, "apply"))
        self._layer_panel.mask_remove.connect(lambda p: self._on_mask_op(p, "remove"))
        self._layer_panel.offset_reset.connect(self._on_layer_center)

        # ── Altri segnali ─────────────────────────────────────────────
        self._filter_panel.filters_changed.connect(self._on_filter_preview)
        self._filter_panel.apply_clicked.connect(self._apply_filters)
        self._filter_panel.reset_clicked.connect(self._reset_filters)

        self._transform_panel.rotate_requested.connect(self._on_rotate)
        self._transform_panel.flip_requested.connect(self._on_flip)
        self._transform_panel.crop_start.connect(self._canvas.start_crop_mode)
        self._transform_panel.crop_confirm.connect(
            lambda: self._canvas.end_crop_mode(confirm=True))
        self._transform_panel.crop_cancel.connect(
            lambda: self._canvas.end_crop_mode(confirm=False))

        self._adjustment_panel.add_requested.connect(self._on_adj_add)
        self._adjustment_panel.update_requested.connect(self._on_adj_update)
        self._free_transform_panel.preview_requested.connect(self._on_ft_preview)
        self._free_transform_panel.apply_requested.connect(self._on_ft_apply)
        self._free_transform_panel.reset_requested.connect(self._on_ft_reset)
        self._export_panel.export_requested.connect(self._on_export)

        self._build_menu()
        self._build_toolbar()
        self._build_statusbar()
        self._update_actions()

        # Timer marching ants (80ms → ~12fps animazione)
        from PyQt6.QtCore import QTimer
        self._ants_timer = QTimer(self)
        self._ants_timer.setInterval(80)
        self._ants_timer.timeout.connect(self._canvas.tick_ants)
        self._ants_timer.start()

    # ── Costruzione UI ────────────────────────────────────────────────

    def _build_menu(self) -> None:
        mb = self.menuBar(); assert mb is not None

        file_menu = mb.addMenu("&File")
        self._act_open        = self._action("&Apri…",                    QKeySequence.StandardKey.Open,   self._on_open)
        self._act_import      = self._action("&Importa layer…",           QKeySequence("Ctrl+Shift+O"),    self._on_import_as_layer)
        self._act_proj_open   = self._action("Apri &progetto…",           QKeySequence("Ctrl+Shift+P"),    self._on_project_open)
        self._act_proj_save   = self._action("Sal&va progetto",           QKeySequence("Ctrl+Shift+S"),    self._on_project_save)
        self._act_proj_saveas = self._action("Salva progetto &con nome…", None,                            self._on_project_save_as)
        self._act_ora_open    = self._action("Apri OpenRaster (.ora)…",   QKeySequence("Ctrl+Alt+O"),      self._on_ora_open)
        self._act_ora_save    = self._action("Esporta OpenRaster (.ora)…",None,                            self._on_ora_save)
        self._act_save        = self._action("&Salva (piatto)",           QKeySequence.StandardKey.Save,   self._on_save)
        self._act_save_as     = self._action("Salva piatto &con nome…",   QKeySequence.StandardKey.SaveAs, self._on_save_as)
        self._act_close       = self._action("&Chiudi",                   QKeySequence("Ctrl+W"),          self._on_close_image)
        act_quit              = self._action("&Esci",                     QKeySequence.StandardKey.Quit,   self.close)
        for a in (self._act_open, self._act_import, self._act_proj_open, None,
                  self._act_proj_save, self._act_proj_saveas, None,
                  self._act_ora_open, self._act_ora_save, None,
                  self._act_save, self._act_save_as, None,
                  self._act_close, None, act_quit):
            file_menu.addAction(a) if a else file_menu.addSeparator()

        edit_menu = mb.addMenu("&Modifica")
        self._act_undo    = self._action("&Annulla",            QKeySequence.StandardKey.Undo,  self._on_undo)
        self._act_redo    = self._action("&Ripeti",             QKeySequence.StandardKey.Redo,  self._on_redo)
        self._act_copy    = self._action("&Copia",             QKeySequence.StandardKey.Copy,  self._on_copy)
        self._act_cut     = self._action("&Taglia selezione",  QKeySequence.StandardKey.Cut,   self._on_cut)
        self._act_paste   = self._action("&Incolla come layer", QKeySequence.StandardKey.Paste, self._on_paste)
        self._act_flatten = self._action("Appiattisci layer",   QKeySequence("Ctrl+Shift+F"),   self._on_flatten)
        edit_menu.addAction(self._act_undo)
        edit_menu.addAction(self._act_redo)
        edit_menu.addSeparator()
        edit_menu.addAction(self._act_copy)
        edit_menu.addAction(self._act_cut)
        edit_menu.addAction(self._act_paste)
        edit_menu.addSeparator()
        edit_menu.addAction(self._action("Seleziona &tutto", QKeySequence("Ctrl+A"),
                                         self._on_sel_all))
        edit_menu.addAction(self._action("&Deseleziona", QKeySequence("Ctrl+D"),
                                         self._on_sel_deselect))
        edit_menu.addSeparator()
        edit_menu.addAction(self._act_flatten)

        layer_menu = mb.addMenu("&Layer")
        layer_menu.addAction(self._action("Nuovo layer",             QKeySequence("Ctrl+Shift+N"), self._on_layer_add))
        layer_menu.addAction(self._action("Nuovo gruppo",            QKeySequence("Ctrl+G"),       self._on_create_group))
        layer_menu.addAction(self._action("Raggruppa selezione",     QKeySequence("Ctrl+Alt+G"), lambda: self._on_group_node(self._layer_stack.active_path if self._layer_stack else (0,))))
        layer_menu.addAction(self._action("Scioglie gruppo",         QKeySequence("Ctrl+Shift+U"), lambda: self._on_ungroup(self._layer_stack.active_path if self._layer_stack else (0,))))
        layer_menu.addSeparator()
        layer_menu.addAction(self._action("Appiattisci tutto",       QKeySequence("Ctrl+Shift+F"), self._on_flatten))

        view_menu = mb.addMenu("&Visualizza")
        for a in (
            self._action("&Adatta", QKeySequence("Ctrl+0"), self._canvas.fit_in_view),
            self._action("100%",    QKeySequence("Ctrl+1"), lambda: self._canvas.set_zoom(1.0)),
            None,
            self._action("Zoom +",  QKeySequence.StandardKey.ZoomIn,  self._canvas.zoom_in),
            self._action("Zoom -",  QKeySequence.StandardKey.ZoomOut, self._canvas.zoom_out),
        ):
            view_menu.addAction(a) if a else view_menu.addSeparator()
        view_menu.addSeparator()
        view_menu.addAction(self._dock.toggleViewAction())
        view_menu.addAction(self._history_dock.toggleViewAction())
        view_menu.addSeparator()
        self._act_rulers = self._action("&Righelli",          QKeySequence("Ctrl+Shift+R"), self._toggle_rulers)
        self._act_grid   = self._action("&Griglia",           QKeySequence("Ctrl+Shift+G"), self._toggle_grid)
        self._act_rulers.setCheckable(True)
        self._act_grid.setCheckable(True)
        view_menu.addAction(self._act_rulers)
        view_menu.addAction(self._act_grid)
        # Spaziatura griglia
        view_menu.addAction(self._action("Spaziatura griglia…", None, self._set_grid_spacing))

        help_menu = mb.addMenu("&Aiuto")
        help_menu.addAction(self._action("Info", None, self._on_about))

    def _build_toolbar(self) -> None:
        tb = QToolBar("Principale", self)
        tb.setMovable(False); tb.setIconSize(QSize(20, 20))
        self.addToolBar(tb)
        for a in (self._act_open, self._act_save, None,
                  self._act_undo, self._act_redo, None):
            tb.addAction(a) if a else tb.addSeparator()

        self._act_move = self._action("✥ Move (M)", QKeySequence("M"), self._toggle_move_tool)
        self._act_move.setCheckable(True)
        tb.addAction(self._act_move)
        tb.addSeparator()
        tb.addAction(self._dock.toggleViewAction())

        for key, shortcut in (("brush","B"), ("eraser","E"), ("smudge","U"),
                               ("eyedropper","I"), ("text","T"),
                               ("select_rect","S"), ("select_lasso","L"),
                               ("clone","C"), ("shape","F")):
            a = self._action("", QKeySequence(shortcut),
                             lambda _, k=key: self._drawing_panel.select_tool(k))
            self.addAction(a)

    def _build_statusbar(self) -> None:
        sb = QStatusBar(); self.setStatusBar(sb)
        self._lbl_file = QLabel("Nessuna immagine aperta")
        self._lbl_file.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self._lbl_mode   = QLabel("")
        self._lbl_layers = QLabel("")
        self._lbl_size   = QLabel("")
        self._lbl_pixel  = QLabel("")
        self._lbl_zoom   = QLabel("Zoom: —")
        for lbl in (self._lbl_mode, self._lbl_layers, self._lbl_size,
                    self._lbl_pixel, self._lbl_zoom):
            lbl.setMinimumWidth(120); lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sb.addWidget(self._lbl_file)
        for lbl in (self._lbl_mode, self._lbl_layers, self._lbl_size,
                    self._lbl_pixel, self._lbl_zoom):
            sb.addPermanentWidget(lbl)

    # ── Slot File ─────────────────────────────────────────────────────

    def _on_open(self) -> None:
        if not self._check_unsaved(): return
        raw_exts = " ".join(f"*{e}" for e in sorted(RAW_EXTENSIONS))
        path, _ = QFileDialog.getOpenFileName(self, "Apri immagine", "",
            f"Tutti i supportati (*.png *.jpg *.jpeg *.webp *.tiff *.tif *.bmp *.gif {raw_exts});;"
            f"Immagini (*.png *.jpg *.jpeg *.webp *.tiff *.tif *.bmp *.gif);;"
            f"File RAW ({raw_exts})")
        if not path: return
        try:
            if is_raw(path):
                array = load_raw(path)
            else:
                array = load_image(path)
        except ImportError as exc:
            QMessageBox.warning(self, "rawpy non installato",
                str(exc) + "\n\nPuoi installarlo con:\n  pip install rawpy")
            return
        except Exception as exc:
            QMessageBox.critical(self, "Errore", str(exc)); return
        self._current_path = FilePath(path)
        self._layer_stack = LayerStack.from_image(array, FilePath(path).stem)
        self._history.clear()
        self._history.push(self._layer_stack.snapshot())
        self._canvas.set_image(self._layer_stack.composite())
        self._unsaved_changes = False
        self._filters = FilterStack(); self._preview = None
        self._filter_panel._reset()
        self._set_selection(None)
        self._clone_source_canvas = None
        self._ft_original_data = None
        self._layer_panel.refresh_from_stack(self._layer_stack)
        self._update_title(); self._update_size_label(array)
        self._update_layer_label(); self._update_actions()

    def _on_import_as_layer(self) -> None:
        if self._layer_stack is None:
            QMessageBox.information(self, "Import", "Apri prima un'immagine."); return
        path, _ = QFileDialog.getOpenFileName(self, "Importa come layer", "",
            "Immagini (*.png *.jpg *.jpeg *.webp *.tiff *.tif *.bmp *.gif)")
        if not path: return
        try:
            array = load_image(path)
        except Exception as exc:
            QMessageBox.critical(self, "Errore", str(exc)); return
        cw, ch = self._layer_stack.canvas_size
        lh, lw = array.shape[:2]
        ox, oy = max(0, (cw - lw) // 2), max(0, (ch - lh) // 2)
        self._layer_stack.add_layer_from_array(array, FilePath(path).stem,
                                                offset_x=ox, offset_y=oy)
        self._commit()

    def _on_save(self) -> None:
        if self._current_path is None: self._on_save_as()
        else: self._save_to(self._current_path)

    def _on_save_as(self) -> None:
        if self._layer_stack is None: return
        WRITE = "PNG (*.png);;JPEG (*.jpg *.jpeg);;WebP (*.webp);;TIFF (*.tiff);;BMP (*.bmp)"
        path, _ = QFileDialog.getSaveFileName(self, "Salva", str(self._current_path or ""), WRITE)
        if not path: return
        self._current_path = FilePath(path); self._save_to(self._current_path)

    def _save_to(self, path: FilePath) -> None:
        if self._layer_stack is None: return
        try:
            save_image(self._layer_stack.composite(), path)
            self._unsaved_changes = False; self._update_title()
            self.statusBar().showMessage(f"Salvato: {path.name}", 3000)  # type: ignore[union-attr]
        except Exception as exc:
            QMessageBox.critical(self, "Errore", str(exc))

    def _on_close_image(self) -> None:
        if not self._check_unsaved(): return
        self._canvas.clear(); self._history.clear()
        self._layer_stack = None; self._current_path = None; self._project_path = None
        self._unsaved_changes = False
        for lbl in (self._lbl_file, self._lbl_size, self._lbl_pixel,
                    self._lbl_zoom, self._lbl_layers, self._lbl_mode):
            lbl.setText("")
        self._lbl_file.setText("Nessuna immagine aperta")
        self.setWindowTitle("Image Suite"); self._update_actions()

    # ── Slot Progetto (.isp) ──────────────────────────────────────────

    def _on_project_open(self) -> None:
        if not self._check_unsaved(): return
        path, _ = QFileDialog.getOpenFileName(
            self, "Apri progetto", "",
            "Image Suite Project (*.isp);;Tutti i file (*)")
        if not path: return
        try:
            stack = load_project(path)
        except Exception as exc:
            QMessageBox.critical(self, "Errore apertura progetto", str(exc)); return
        self._layer_stack = stack
        self._project_path = FilePath(path)
        self._current_path = None
        self._history.clear()
        self._history.push(stack.snapshot())
        self._canvas.set_image(stack.composite())
        self._unsaved_changes = False
        self._filters = FilterStack(); self._preview = None
        self._filter_panel._reset()
        self._layer_panel.refresh_from_stack(stack)
        self._update_title(); self._update_actions(); self._update_layer_label()
        cw, ch = stack.canvas_size
        self._lbl_size.setText(f"{cw} × {ch} px")

    def _on_project_save(self) -> None:
        if self._project_path is None:
            self._on_project_save_as()
        else:
            self._save_project_to(self._project_path)

    def _on_project_save_as(self) -> None:
        if self._layer_stack is None: return
        default = str(self._project_path or
                      (self._current_path.with_suffix(".isp") if self._current_path else ""))
        path, _ = QFileDialog.getSaveFileName(
            self, "Salva progetto", default,
            "Image Suite Project (*.isp)")
        if not path: return
        if not path.endswith(".isp"):
            path += ".isp"
        self._project_path = FilePath(path)
        self._save_project_to(self._project_path)

    def _save_project_to(self, path: FilePath) -> None:
        if self._layer_stack is None: return
        try:
            save_project(self._layer_stack, path)
            self._unsaved_changes = False
            self._update_title()
            self.statusBar().showMessage(f"Progetto salvato: {path.name}", 3000)  # type: ignore[union-attr]
        except Exception as exc:
            QMessageBox.critical(self, "Errore salvataggio progetto", str(exc))

    # ── OpenRaster (.ora) ─────────────────────────────────────────────

    def _on_ora_open(self) -> None:
        if not self._check_unsaved(): return
        path, _ = QFileDialog.getOpenFileName(
            self, "Apri OpenRaster", "",
            "OpenRaster (*.ora);;Tutti i file (*)")
        if not path: return
        try:
            stack = load_ora(path)
        except Exception as exc:
            QMessageBox.critical(self, "Errore apertura ORA", str(exc)); return
        self._layer_stack = stack
        self._project_path = None
        self._current_path = FilePath(path)
        self._history.clear()
        self._history.push(stack.snapshot())
        self._canvas.set_image(stack.composite())
        self._unsaved_changes = False
        self._filters = FilterStack(); self._preview = None
        self._filter_panel._reset()
        self._layer_panel.refresh_from_stack(stack)
        self._update_title(); self._update_actions(); self._update_layer_label()
        cw, ch = stack.canvas_size
        self._lbl_size.setText(f"{cw} × {ch} px")

    def _on_ora_save(self) -> None:
        if self._layer_stack is None: return
        default = ""
        if self._current_path:
            default = str(self._current_path.with_suffix(".ora"))
        elif self._project_path:
            default = str(self._project_path.with_suffix(".ora"))
        path, _ = QFileDialog.getSaveFileName(
            self, "Esporta OpenRaster", default,
            "OpenRaster (*.ora)")
        if not path: return
        if not path.endswith(".ora"):
            path += ".ora"
        try:
            save_ora(self._layer_stack, path)
            self.statusBar().showMessage(f"Esportato: {FilePath(path).name}", 3000)  # type: ignore[union-attr]
        except Exception as exc:
            QMessageBox.critical(self, "Errore esportazione ORA", str(exc))

    # ── Clipboard ─────────────────────────────────────────────────────

    def _on_copy(self) -> None:
        """Copia la selezione (se attiva) o il layer intero nella clipboard di sistema."""
        if self._layer_stack is None: return
        layer = self._layer_stack.active

        if self._sel_mask is not None and self._sel_mask.any():
            # Copia la regione selezionata
            patch = copy_region(layer.data, self._sel_mask,
                                layer.offset_x, layer.offset_y)
            self._sel_clipboard = patch
            arr = patch
            msg = f"Selezione copiata: {patch.shape[1]}×{patch.shape[0]} px"
        else:
            # Copia il layer intero
            arr = layer.data
            self._sel_clipboard = arr.copy()
            msg = f"Layer «{layer.name}» copiato negli appunti."

        h, w = arr.shape[:2]
        qimg = QImage(arr.tobytes(), w, h, w * 4, QImage.Format.Format_RGBA8888)
        QApplication.clipboard().setImage(qimg)
        self.statusBar().showMessage(msg, 2500)  # type: ignore[union-attr]

    def _on_cut(self) -> None:
        """Ctrl+X: taglia la selezione se attiva, altrimenti non fa nulla."""
        if self._sel_mask is not None and self._sel_mask.any():
            self._on_sel_cut()

    # def _on_paste(self):
    #     # from PyQt6.QtWidgets import QApplication
    #     # from PyQt6.QtGui import QImage
    #     # import numpy as np

    #     clipboard = QApplication.clipboard()

    #     if not clipboard.mimeData().hasImage():
    #         return

    #     qimg = clipboard.image()

    #     if qimg.isNull():
    #         return

    #     # Converti in formato RGBA
    #     qimg = qimg.convertToFormat(QImage.Format.Format_RGBA8888)

    #     # Ottieni il puntatore ai dati
    #     ptr = qimg.constBits()
    #     ptr.setsize(qimg.sizeInBytes())

    #     # Converti in numpy array gestendo il padding delle righe
    #     arr = np.frombuffer(ptr, np.uint8).reshape(
    #         qimg.height(),
    #         qimg.bytesPerLine() // 4,
    #         4
    #     )

    #     # Rimuovi eventuale padding
    #     arr = arr[:, :qimg.width(), :].copy()

    #     # Incolla nel layer
    #     self._paste_array(arr, "Incollato")
    def _on_paste(self):
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtGui import QImage
        from PyQt6.QtCore import QUrl
        import numpy as np

        clipboard = QApplication.clipboard()
        mime = clipboard.mimeData()

        qimg = None

        # 1️⃣ immagine diretta (browser, screenshot tool)
        if mime.hasImage():
            qimg = clipboard.image()

        # 2️⃣ immagine PNG nei mimeData
        elif mime.hasFormat("image/png"):
            data = mime.data("image/png")
            qimg = QImage()
            qimg.loadFromData(data)

        # 3️⃣ file copiato dal file manager
        elif mime.hasUrls():
            urls = mime.urls()
            if urls:
                path = urls[0].toLocalFile()
                if path:
                    qimg = QImage(path)

        if qimg is None or qimg.isNull():
            return

        # converti formato
        qimg = qimg.convertToFormat(QImage.Format.Format_RGBA8888)

        ptr = qimg.constBits()
        ptr.setsize(qimg.sizeInBytes())

        arr = np.frombuffer(ptr, np.uint8).reshape(
            qimg.height(),
            qimg.bytesPerLine() // 4,
            4
        )

        arr = arr[:, :qimg.width(), :].copy()

        self._paste_array(arr, "Paste")

    def _paste_array(self, arr: np.ndarray, name: str) -> None:
        """Incolla un array RGBA come nuovo layer centrato nel canvas."""
        if self._layer_stack is None:
            # Nessun progetto aperto: crea uno nuovo dal file droppato
            self._layer_stack = LayerStack.from_image(arr, name)
            self._history.clear()
            self._history.push(self._layer_stack.snapshot())
            self._canvas.set_image(self._layer_stack.composite())
            self._unsaved_changes = False
            self._layer_panel.refresh_from_stack(self._layer_stack)
            self._update_title(); self._update_actions(); self._update_layer_label()
            h, w = arr.shape[:2]
            self._lbl_size.setText(f"{w} × {h} px")
            return
        cw, ch = self._layer_stack.canvas_size
        lh, lw = arr.shape[:2]
        ox = max(0, (cw - lw) // 2)
        oy = max(0, (ch - lh) // 2)
        self._layer_stack.add_layer_from_array(arr, name, offset_x=ox, offset_y=oy)
        self._commit()
        self.statusBar().showMessage("Immagine incollata come nuovo layer.", 2500)  # type: ignore[union-attr]

    def _on_file_dropped(self, path: str) -> None:
        """File locale droppato sul canvas dal file manager."""
        try:
            arr = load_image(path)
        except Exception as exc:
            QMessageBox.critical(self, "Errore drop", str(exc)); return
        name = FilePath(path).stem
        # Se nessun progetto è aperto, apri il file come progetto principale
        if self._layer_stack is None:
            if not self._check_unsaved(): return
            self._layer_stack = LayerStack.from_image(arr, name)
            self._current_path = FilePath(path)
            self._project_path = None
            self._history.clear()
            self._history.push(self._layer_stack.snapshot())
            self._canvas.set_image(self._layer_stack.composite())
            self._unsaved_changes = False
            self._filters = FilterStack(); self._preview = None
            self._filter_panel._reset()
            self._layer_panel.refresh_from_stack(self._layer_stack)
            self._update_title(); self._update_size_label(arr)
            self._update_layer_label(); self._update_actions()
        else:
            self._paste_array(arr, name)

    def _on_image_dropped(self, arr: np.ndarray) -> None:
        """Immagine raw (da altra app) droppata sul canvas."""
        self._paste_array(arr, "Immagine incollata")

    # ── Undo / Redo ───────────────────────────────────────────────────

    def _on_undo(self) -> None:
        snap = self._history.undo()
        if snap:
            self._layer_stack = LayerStack.from_snapshot(snap)
            self._reset_filter_state()
            self._canvas.update_image(self._layer_stack.composite())
            self._layer_panel.refresh_from_stack(self._layer_stack)
            self._mark_changed(); self._update_layer_label()
        self._update_actions()

    def _on_redo(self) -> None:
        snap = self._history.redo()
        if snap:
            self._layer_stack = LayerStack.from_snapshot(snap)
            self._reset_filter_state()
            self._canvas.update_image(self._layer_stack.composite())
            self._layer_panel.refresh_from_stack(self._layer_stack)
            self._mark_changed(); self._update_layer_label()
        self._update_actions()

    # ── Filtri ────────────────────────────────────────────────────────

    def _on_filter_preview(self, values: dict) -> None:
        if self._layer_stack is None: return
        self._filters.update(**values)
        base = self._layer_stack.active.data
        preview = self._filters.render(base)
        self._preview = preview
        tmp = self._layer_stack.composite_with_override(
            self._layer_stack.active_path, preview)
        self._canvas.update_image(tmp)

    def _apply_filters(self) -> None:
        if self._preview is None or self._layer_stack is None: return
        self._layer_stack.set_active_data(self._preview)
        self._preview = None; self._commit()

    def _reset_filters(self) -> None:
        if self._layer_stack is None: return
        self._filters = FilterStack(); self._preview = None
        self._canvas.update_image(self._layer_stack.composite())

    # ── Strumenti disegno ─────────────────────────────────────────────

    def _on_draw_tool_changed(self, tool: str) -> None:
        prev = self._prev_draw_tool; self._prev_draw_tool = tool
        if prev == "text" and tool != "text":
            self._apply_text(); self._canvas.clear_text_anchor()
        if prev in ("select_rect","select_ellipse","select_lasso") and \
                tool not in ("select_rect","select_ellipse","select_lasso"):
            self._canvas.cancel_selection_mode()
        if prev == "clone" and tool != "clone":
            self._canvas.end_clone_mode()
        if prev == "magic_wand" and tool != "magic_wand":
            self._cancel_wand_thread()
        if prev == "shape" and tool != "shape":
            self._current_shape = None
            self._canvas.set_shape(None)
            self._canvas.set_shape_ghost_kind(None)
        if tool == "shape":
            self._canvas.set_shape_ghost_kind(
                _shape_key_to_kind(self._drawing_panel.shape_kind_key))
        if tool and self._canvas.mode == CanvasMode.MOVE:
            self._canvas.set_move_mode(False); self._act_move.setChecked(False)

        is_sel   = tool in ("select_rect", "select_ellipse", "select_lasso")
        is_clone = tool == "clone"
        if is_sel:
            self._canvas.start_selection_mode(tool.replace("select_", ""))
        elif is_clone:
            self._canvas.start_clone_mode()
        else:
            self._canvas.set_draw_mode(tool)

        label_map = {
            "brush":          "🖌 Pennello",
            "eraser":         "⬜ Gomma",
            "smudge":         "💧 Sfumino",
            "eyedropper":     "🔍 Contagocce",
            "text":           "T Testo",
            "select_rect":    "⬜ Selezione rettangolare",
            "select_ellipse": "◯ Selezione ellittica",
            "select_lasso":   "🔗 Lasso",
            "clone":          "🔁 Clone",
            "gradient":       "🌈 Gradiente",
            "heal":           "🩹 Guarigione",
            "magic_wand":     "🪄 Bacchetta magica",
            "shape":          "⬟ Forme vettoriali",
        }
        self._lbl_mode.setText(label_map.get(tool, ""))
        self._update_actions()

    def _on_draw_color_changed(self, _: QColor) -> None:
        self._update_text_preview_on_canvas()

    def _on_draw_size_changed(self, size: int) -> None:
        self._canvas.set_brush_size(size)

    def _on_draw_hardness_changed(self, _: int) -> None: pass
    def _on_draw_opacity_changed(self, _: int) -> None: pass

    def _on_stroke_begin(self, x: float, y: float) -> None:
        if self._layer_stack is None: return
        self._draw_bg_cache  = self._layer_stack.composite_without_active()
        self._canvas_img_buf = self._layer_stack.fast_composite_active(self._draw_bg_cache)
        self._canvas.update_image(self._canvas_img_buf)
        self._last_stroke_pt = (x, y)
        if self._drawing_panel.active_tool != "smudge":
            dirty = self._apply_stamp(x, y)
            if dirty and dirty != (0, 0, 0, 0):
                self._flush_dirty(dirty)

    def _on_stroke_move(self, x: float, y: float) -> None:
        if self._layer_stack is None or self._last_stroke_pt is None: return
        tool = self._drawing_panel.active_tool

        if tool == "smudge":
            layer = self._layer_stack.active
            lx = int(x - layer.offset_x)
            ly = int(y - layer.offset_y)
            apply_smudge(layer.data, lx, ly,
                         self._drawing_panel.brush_size,
                         self._drawing_panel.hardness / 100.0)
            r = self._drawing_panel.brush_size // 2 + 2
            self._flush_dirty((lx - r, ly - r, lx + r, ly + r))
        else:
            pts = interpolate_stroke(self._last_stroke_pt, (x, y),
                                     self._drawing_panel.brush_size)
            union = _union_dirty(self._apply_stamp(px, py) for px, py in pts)
            if union:
                self._flush_dirty(union)

        self._last_stroke_pt = (x, y)

    def _flush_dirty(self, dirty_layer: tuple) -> None:
        """Ricomposta solo il patch sporco e aggiorna solo quella regione del canvas."""
        if self._draw_bg_cache is None or self._canvas_img_buf is None: return
        layer = self._layer_stack.active
        lx0, ly0, lx1, ly1 = dirty_layer
        cx0 = int(lx0) + layer.offset_x
        cy0 = int(ly0) + layer.offset_y
        cx1 = int(lx1) + layer.offset_x
        cy1 = int(ly1) + layer.offset_y
        ch, cw = self._draw_bg_cache.shape[:2]
        cx0 = max(0, cx0); cy0 = max(0, cy0)
        cx1 = min(cw, cx1); cy1 = min(ch, cy1)
        if cx0 >= cx1 or cy0 >= cy1: return
        self._layer_stack.fast_composite_patch(
            self._draw_bg_cache, self._canvas_img_buf,
            cx0, cy0, cx1, cy1)
        self._canvas.update_image_patch(
            self._canvas_img_buf, cx0, cy0, cx1, cy1)

    def _on_stroke_end(self) -> None:
        self._last_stroke_pt = None
        self._draw_bg_cache  = None
        self._canvas_img_buf = None
        if self._layer_stack is None: return
        composite = self._layer_stack.composite()
        self._canvas.update_image(composite)
        self._history.push(self._layer_stack.snapshot(), composite)
        self._mark_changed(); self._update_actions()
        self._refresh_history_panel()

    def _apply_stamp(self, x: float, y: float) -> tuple:
        """Applica un dab e restituisce la dirty rect in coordinate LAYER."""
        if self._layer_stack is None: return (0, 0, 0, 0)
        layer = self._layer_stack.active
        c = self._drawing_panel.current_color()
        return paint_stamp(
            data=layer.data,
            cx=x - layer.offset_x, cy=y - layer.offset_y,
            color=(c.red(), c.green(), c.blue(), c.alpha()),
            size=self._drawing_panel.brush_size,
            hardness=self._drawing_panel.hardness,
            opacity=self._drawing_panel.opacity,
            mode=self._drawing_panel.active_tool)

    def _on_eyedropper_pick(self, x: int, y: int) -> None:
        arr = self._canvas.get_image()
        if arr is None: return
        h, w = arr.shape[:2]
        if 0 <= x < w and 0 <= y < h:
            r, g, b, a = arr[y, x]
            self._drawing_panel.set_color(QColor(int(r), int(g), int(b), int(a)))
            self.statusBar().showMessage(f"Colore: RGBA({r},{g},{b},{a})", 2000)  # type: ignore[union-attr]

    def _on_text_anchor_changed(self, x: float, y: float) -> None:
        self._update_text_preview_on_canvas()

    def _on_text_params_changed(self) -> None:
        self._update_text_preview_on_canvas()

    def _update_text_preview_on_canvas(self) -> None:
        dp = self._drawing_panel
        ec = dp.text_effect_color
        self._canvas.set_text_preview(
            text=dp.text_content,
            font_family=dp.font_family,
            font_size=dp.font_size_px,
            bold=dp.font_bold,
            italic=dp.font_italic,
            color=dp.current_color(),
            opacity=dp.text_opacity,
            effect=dp.text_effect,
            effect_color=QColor(ec[0], ec[1], ec[2], ec[3]),
            outline_width=dp.text_outline_width,
        )

    def _apply_text(self) -> None:
        anchor = self._canvas.get_text_anchor()
        if anchor is None or self._layer_stack is None: return
        dp = self._drawing_panel
        text = dp.text_content.strip()
        if not text: return
        layer = self._layer_stack.active
        c  = dp.current_color()
        ec = dp.text_effect_color
        draw_text(
            data=layer.data,
            x=int(anchor[0] - layer.offset_x),
            y=int(anchor[1] - layer.offset_y),
            text=text,
            color_rgba=(c.red(), c.green(), c.blue(), c.alpha()),
            font_family=dp.font_family,
            font_size=dp.font_size_px,
            bold=dp.font_bold,
            italic=dp.font_italic,
            opacity=dp.text_opacity,
            effect=dp.text_effect,
            effect_color=ec,
            outline_width=dp.text_outline_width,
        )
        self._canvas.clear_text_anchor()
        self._commit()
        self.statusBar().showMessage(f"Testo \"{text[:30]}\" applicato.", 2500)  # type: ignore[union-attr]

    # ── Trasformazioni ────────────────────────────────────────────────

    def _on_crop_committed(self, x: int, y: int, w: int, h: int) -> None:
        if self._layer_stack is None: return
        img = self._layer_stack.active.data
        ih, iw = img.shape[:2]
        x2, y2 = min(x + w, iw), min(y + h, ih)
        if x2 > x and y2 > y:
            self._layer_stack.set_active_data(img[y:y2, x:x2].copy())
            self._commit()

    def _on_rotate(self, degrees: int) -> None:
        if self._layer_stack is None: return
        k = {90: 3, -90: 1, 180: 2}.get(degrees, 0)
        self._layer_stack.set_active_data(
            np.rot90(self._layer_stack.active.data, k=k).copy()); self._commit()

    def _on_flip(self, horizontal: bool) -> None:
        if self._layer_stack is None: return
        img = self._layer_stack.active.data
        self._layer_stack.set_active_data(
            (np.fliplr(img) if horizontal else np.flipud(img)).copy()); self._commit()

    def _on_flatten(self) -> None:
        if self._layer_stack is None: return
        flat = self._layer_stack.composite()
        self._layer_stack = LayerStack.from_image(flat, "Appiattito"); self._commit()

    # ── Move tool ─────────────────────────────────────────────────────

    def _toggle_move_tool(self) -> None:
        if self._drawing_panel.active_tool:
            self._drawing_panel.select_tool(self._drawing_panel.active_tool)
        enabled = self._canvas.mode != CanvasMode.MOVE
        self._canvas.set_move_mode(enabled)
        self._act_move.setChecked(enabled)
        self._lbl_mode.setText("✥ Move" if enabled else "")

    def _on_layer_moved(self, dx: int, dy: int) -> None:
        if self._layer_stack is None: return
        self._layer_stack.set_active_offset(dx, dy)
        self._canvas.update_image(self._layer_stack.composite())
        self._unsaved_changes = True; self._update_title()

    def _on_layer_center(self, path: tuple) -> None:
        if self._layer_stack is None: return
        node = self._layer_stack.get_node(path)
        if not isinstance(node, Layer): return
        cw, ch = self._layer_stack.canvas_size
        lh, lw = node.data.shape[:2]
        node.offset_x = max(0, (cw - lw) // 2)
        node.offset_y = max(0, (ch - lh) // 2)
        self._commit()

    # ── Layer panel ───────────────────────────────────────────────────

    def _on_layer_active_changed(self, path: tuple) -> None:
        if self._layer_stack is None: return
        self._layer_stack.set_active_path(path)
        self._layer_panel.refresh_from_stack(self._layer_stack)
        self._update_layer_label()
        # Se il nodo è un AdjustmentLayer carica i parametri nel pannello
        try:
            node = self._layer_stack.get_node(tuple(path))
            if isinstance(node, AdjustmentLayer):
                self._adjustment_panel.load_adjustment(node.kind, node.params)
            else:
                self._adjustment_panel.reset_ui()
        except Exception:
            pass

    def _on_layer_visibility_toggled(self, path: tuple) -> None:
        if self._layer_stack is None: return
        node = self._layer_stack.get_node(path)
        node.visible = not node.visible
        self._canvas.update_image(self._layer_stack.composite())
        self._layer_panel.refresh_from_stack(self._layer_stack)

    def _on_layer_opacity_changed(self, path: tuple, opacity: float) -> None:
        if self._layer_stack is None: return
        self._layer_stack.get_node(path).opacity = opacity
        self._canvas.update_image(self._layer_stack.composite())

    def _on_layer_blend_changed(self, path: tuple, mode: str) -> None:
        if self._layer_stack is None: return
        self._layer_stack.get_node(path).blend_mode = mode
        self._canvas.update_image(self._layer_stack.composite())

    def _on_layer_add(self) -> None:
        if self._layer_stack is None: return
        self._layer_stack.add_layer(); self._commit()

    def _on_layer_delete(self, path: tuple) -> None:
        if self._layer_stack is None: return
        if not self._layer_stack.remove_node(path):
            QMessageBox.information(self, "Layer", "Deve restare almeno un layer."); return
        self._commit()

    def _on_layer_duplicate(self, path: tuple) -> None:
        if self._layer_stack is None: return
        self._layer_stack.duplicate_node(path); self._commit()

    def _on_layer_move_up(self, path: tuple) -> None:
        if self._layer_stack is None: return
        self._layer_stack.move_up(path); self._commit()

    def _on_layer_move_down(self, path: tuple) -> None:
        if self._layer_stack is None: return
        self._layer_stack.move_down(path); self._commit()

    def _on_layer_rename(self, path: tuple, name: str) -> None:
        if self._layer_stack is None: return
        self._layer_stack.get_node(path).name = name
        self._layer_panel.refresh_from_stack(self._layer_stack)

    # ── Operazioni gruppo ─────────────────────────────────────────────

    def _on_create_group(self) -> None:
        if self._layer_stack is None: return
        name, ok = QInputDialog.getText(self, "Nuovo gruppo", "Nome gruppo:", text="Gruppo")
        if not ok or not name.strip(): return
        self._layer_stack.create_group(name.strip())
        self._commit()

    def _on_group_node(self, path: tuple) -> None:
        if self._layer_stack is None: return
        name, ok = QInputDialog.getText(self, "Raggruppa", "Nome gruppo:", text="Gruppo")
        if not ok or not name.strip(): return
        self._layer_stack.group_node(path, name.strip())
        self._commit()

    def _on_ungroup(self, path: tuple) -> None:
        if self._layer_stack is None: return
        node = self._layer_stack.get_node(path)
        if not isinstance(node, LayerGroup):
            QMessageBox.information(self, "Ungroup", "Il nodo selezionato non è un gruppo.")
            return
        self._layer_stack.ungroup(path); self._commit()

    def _on_move_into_group(self, node_path: tuple, group_path: tuple) -> None:
        if self._layer_stack is None: return
        if not self._layer_stack.move_into_group(node_path, group_path):
            QMessageBox.information(self, "Sposta", "Operazione non consentita.")
            return
        self._commit()

    def _on_move_out_of_group(self, path: tuple) -> None:
        if self._layer_stack is None: return
        if not self._layer_stack.move_out_of_group(path):
            QMessageBox.information(self, "Sposta", "Il layer è già al livello radice.")
            return
        self._commit()

    # ── Maschera ──────────────────────────────────────────────────────

    def _on_mask_op(self, path: tuple, op: str) -> None:
        if self._layer_stack is None: return
        node = self._layer_stack.get_node(path)
        if not isinstance(node, Layer):
            QMessageBox.information(self, "Maschera", "Le maschere si applicano solo ai layer foglia.")
            return
        if op == "white":   node.add_mask(255)
        elif op == "black": node.add_mask(0)
        elif op == "invert":
            if node.mask is None:
                QMessageBox.information(self, "Maschera", "Nessuna maschera presente."); return
            node.invert_mask()
        elif op == "apply":
            if node.mask is None:
                QMessageBox.information(self, "Maschera", "Nessuna maschera presente."); return
            node.apply_mask_permanently()
        elif op == "remove": node.remove_mask()
        self._commit()

    # ── Export ────────────────────────────────────────────────────────

    def _on_export(self, fmt_key: str, quality: int, resize_pct: int) -> None:
        if self._layer_stack is None:
            QMessageBox.warning(self, "Esporta", "Nessuna immagine aperta."); return
        _, ext, _, _ = FORMATS[fmt_key]
        default_name = (self._current_path.stem if self._current_path else "immagine") + ext
        path, _ = QFileDialog.getSaveFileName(self, f"Esporta come {fmt_key}",
                                              default_name, f"{fmt_key} (*{ext})")
        if not path: return
        export_array = self._layer_stack.composite()
        if resize_pct != 100:
            import cv2
            h, w = export_array.shape[:2]
            nw, nh = max(1, int(w * resize_pct / 100)), max(1, int(h * resize_pct / 100))
            export_array = cv2.resize(export_array, (nw, nh),
                interpolation=cv2.INTER_LANCZOS4 if resize_pct < 100 else cv2.INTER_CUBIC)
        try:
            save_image(export_array, path, quality=quality)
            self.statusBar().showMessage(f"Esportato: {FilePath(path).name}", 4000)  # type: ignore[union-attr]
        except Exception as exc:
            QMessageBox.critical(self, "Errore esportazione", str(exc))

    # ── Canvas ────────────────────────────────────────────────────────

    def _on_zoom_changed(self, f: float) -> None:
        self._lbl_zoom.setText(f"Zoom: {f * 100:.0f}%")

    def _on_pixel_hovered(self, x: int, y: int) -> None:
        arr = self._canvas.get_image()
        if arr is None: return
        r, g, b, a = arr[y, x]
        self._lbl_pixel.setText(f"({x}, {y})  RGBA({r},{g},{b},{a})")

    # ── Slot Scontornamento ──────────────────────────────────────────

    def _on_rembg_requested(self, model: str, alpha_matting: bool,
                             fg_thresh: int, bg_thresh: int, erode: int) -> None:
        if self._layer_stack is None: return

        # Guard: thread ancora attivo?
        if self._rembg_thread is not None:
            QMessageBox.information(self, "Attendere",
                "Rimozione sfondo già in corso. Attendere il completamento.")
            return

        array = self._layer_stack.active.data.copy()

        self._rembg_worker = _RembgWorker(array, model, alpha_matting,
                                           fg_thresh, bg_thresh, erode)
        self._rembg_thread = QThread(self)
        self._rembg_worker.moveToThread(self._rembg_thread)

        self._rembg_thread.started.connect(self._rembg_worker.run)
        self._rembg_worker.done.connect(self._on_rembg_done)
        self._rembg_worker.error.connect(self._on_rembg_error)
        self._rembg_worker.done.connect(self._rembg_thread.quit)
        self._rembg_worker.error.connect(self._rembg_thread.quit)
        # Quando il thread finisce: distruggi gli oggetti C++ e azzera i riferimenti Python
        self._rembg_thread.finished.connect(self._rembg_worker.deleteLater)
        self._rembg_thread.finished.connect(self._rembg_thread.deleteLater)
        self._rembg_thread.finished.connect(self._rembg_cleanup)

        self._rembg_thread.start()
        self.statusBar().showMessage("⏳ Rimozione sfondo in corso…")  # type: ignore[union-attr]

    def _rembg_cleanup(self) -> None:
        """Azzerata dopo che il thread C++ è stato distrutto via deleteLater."""
        self._rembg_thread = None
        self._rembg_worker = None

    def _on_rembg_done(self, result) -> None:
        if self._layer_stack is None: return
        self._layer_stack.set_active_data(result)
        self._commit()
        self.statusBar().showMessage("✓ Sfondo rimosso.", 3000)  # type: ignore[union-attr]

    def _on_rembg_error(self, msg: str) -> None:
        QMessageBox.critical(self, "Errore rembg", msg)
        self.statusBar().showMessage("Errore rimozione sfondo.", 3000)  # type: ignore[union-attr]

    def _on_grabcut_start(self) -> None:
        if self._layer_stack is None:
            QMessageBox.information(self, "GrabCut", "Apri prima un'immagine.")
            self._matting_panel.set_grabcut_active(False); return
        # Disattiva qualsiasi drawing tool attivo prima di entrare in GrabCut
        if self._drawing_panel.active_tool:
            self._drawing_panel.select_tool(self._drawing_panel.active_tool)  # toggle off
        if self._canvas.mode == CanvasMode.MOVE:
            self._canvas.set_move_mode(False)
            self._act_move.setChecked(False)
        self._canvas.start_grabcut_mode()

    def _on_grabcut_cancel(self) -> None:
        self._canvas.end_grabcut_mode(confirm=False)

    def _on_grabcut_apply(self, iterations: int) -> None:
        # Segnale dal pannello: chiede conferma → il canvas emette grabcut_rect_committed
        self._canvas.end_grabcut_mode(confirm=True)
        # _on_grabcut_rect verrà chiamato se il canvas ha un rect valido

    def _on_grabcut_rect(self, x: int, y: int, w: int, h: int) -> None:
        if self._layer_stack is None: return
        iterations = self._matting_panel._spin_iter.value()
        try:
            from core.matting import grabcut_rect
            result = grabcut_rect(self._layer_stack.active.data,
                                  (x, y, w, h), iterations)
            self._layer_stack.set_active_data(result)
            self._commit()
            self.statusBar().showMessage("✓ GrabCut applicato.", 3000)  # type: ignore[union-attr]
        except Exception as exc:
            QMessageBox.critical(self, "Errore GrabCut", str(exc))

    def _on_refine_mask(self, erode: int, dilate: int, blur: int) -> None:
        if self._layer_stack is None: return
        try:
            from core.matting import refine_mask_morphology
            result = refine_mask_morphology(self._layer_stack.active.data,
                                            erode, dilate, blur)
            self._layer_stack.set_active_data(result)
            self._commit()
            self.statusBar().showMessage("✓ Bordi rifiniti.", 2500)  # type: ignore[union-attr]
        except Exception as exc:
            QMessageBox.critical(self, "Errore rifinitura", str(exc))

    # ── Slot Selezione ───────────────────────────────────────────────

    def _set_selection(self, mask) -> None:
        """Imposta la maschera di selezione e aggiorna il canvas."""
        self._sel_mask = mask
        self._canvas.set_selection_mask(mask)

    def _on_selection_rect(self, x: int, y: int, w: int, h: int) -> None:
        if self._layer_stack is None: return
        cw, ch = self._layer_stack.canvas_size
        mask = mask_from_rect(ch, cw, x, y, w, h)
        self._set_selection(mask)
        self.statusBar().showMessage(f"Selezione rettangolare: {w}×{h} px", 2000)  # type: ignore[union-attr]

    def _on_selection_ellipse(self, x: int, y: int, w: int, h: int) -> None:
        if self._layer_stack is None: return
        cw, ch = self._layer_stack.canvas_size
        mask = mask_from_ellipse(ch, cw, x, y, w, h)
        self._set_selection(mask)
        self.statusBar().showMessage(f"Selezione ellittica: {w}×{h} px", 2000)  # type: ignore[union-attr]

    def _on_selection_lasso(self, pts: list) -> None:
        if self._layer_stack is None: return
        cw, ch = self._layer_stack.canvas_size
        mask = mask_from_lasso(ch, cw, pts)
        self._set_selection(mask)
        self.statusBar().showMessage(f"Selezione lasso: {int(mask.sum())} px", 2000)  # type: ignore[union-attr]

    def _on_selection_cancelled(self) -> None:
        self._set_selection(None)

    def _on_sel_all(self) -> None:
        if self._layer_stack is None: return
        cw, ch = self._layer_stack.canvas_size
        self._set_selection(select_all(ch, cw))
        self.statusBar().showMessage("Tutto selezionato", 1500)  # type: ignore[union-attr]

    def _on_sel_deselect(self) -> None:
        self._set_selection(None)
        self.statusBar().showMessage("Selezione annullata", 1500)  # type: ignore[union-attr]

    def _on_sel_invert(self) -> None:
        if self._sel_mask is None:
            self._on_sel_all(); return
        self._set_selection(invert_mask(self._sel_mask))

    def _on_sel_feather_changed(self, radius: int) -> None:
        self._sel_feather = radius

    def _on_sel_copy(self) -> None:
        """Copia la regione selezionata del layer attivo negli appunti (interni + sistema)."""
        if self._layer_stack is None or self._sel_mask is None: return
        layer = self._layer_stack.active
        patch = copy_region(layer.data, self._sel_mask,
                            layer.offset_x, layer.offset_y)
        self._sel_clipboard = patch
        # Anche nella clipboard di sistema per paste esterno e Ctrl+V
        h, w = patch.shape[:2]
        qimg = QImage(patch.tobytes(), w, h, w * 4, QImage.Format.Format_RGBA8888)
        QApplication.clipboard().setImage(qimg)
        self.statusBar().showMessage(  # type: ignore[union-attr]
            f"Selezione copiata: {w}×{h} px", 2000)

    def _on_sel_cut(self) -> None:
        """Taglia: copia nella clipboard + cancella la regione del layer attivo."""
        if self._layer_stack is None or self._sel_mask is None: return
        layer = self._layer_stack.active
        patch = copy_region(layer.data, self._sel_mask,
                            layer.offset_x, layer.offset_y)
        self._sel_clipboard = patch
        # Anche nella clipboard di sistema
        h, w = patch.shape[:2]
        qimg = QImage(patch.tobytes(), w, h, w * 4, QImage.Format.Format_RGBA8888)
        QApplication.clipboard().setImage(qimg)
        # Cancella dal layer
        new_data = cut_region(layer.data, self._sel_mask,
                              layer.offset_x, layer.offset_y)
        self._layer_stack.set_active_data(new_data)
        self._set_selection(None)
        self._commit()
        self.statusBar().showMessage(f"Tagliato: {w}×{h} px", 2000)  # type: ignore[union-attr]

    def _on_sel_fill(self) -> None:
        """Riempie la selezione con il colore corrente del pannello disegno."""
        if self._layer_stack is None or self._sel_mask is None: return
        c = self._drawing_panel.current_color()
        color = (c.red(), c.green(), c.blue(), c.alpha())
        layer = self._layer_stack.active
        new_data = fill_region(layer.data, self._sel_mask, color,
                               layer.offset_x, layer.offset_y,
                               feather_radius=self._sel_feather)
        self._layer_stack.set_active_data(new_data)
        self._commit()
        self.statusBar().showMessage("Regione riempita", 2000)  # type: ignore[union-attr]

    # ── Slot Clone stamp ──────────────────────────────────────────────

    def _on_clone_source_set(self, cx: float, cy: float) -> None:
        """Salva la sorgente in coordinate canvas."""
        self._clone_source_canvas = (cx, cy)
        self._clone_last_src = None
        self._clone_last_dst = None
        self.statusBar().showMessage(  # type: ignore[union-attr]
            f"Sorgente clone impostata a ({int(cx)}, {int(cy)})", 2000)

    def _on_clone_begin(self, dx: float, dy: float) -> None:
        if self._layer_stack is None or self._clone_source_canvas is None: return
        self._clone_last_dst = (dx, dy)
        self._clone_last_src = self._clone_source_canvas
        self._apply_clone_dab(dx, dy)

    def _on_clone_move(self, dx: float, dy: float) -> None:
        if self._layer_stack is None or self._clone_last_dst is None: return
        # Mantieni lo stesso offset sorgente-destinazione stabilito al primo click
        if self._clone_source_canvas is None: return
        src_cx, src_cy = self._clone_source_canvas
        dst_ox, dst_oy = self._clone_last_dst
        # offset fisso tra dst e src
        off_x = src_cx - dst_ox
        off_y = src_cy - dst_oy

        pts = interpolate_clone_stroke(self._clone_last_dst, (dx, dy),
                                       self._drawing_panel.brush_size)
        layer = self._layer_stack.active
        for px, py in pts:
            sx = px + off_x;  sy = py + off_y
            # Aggiusta la sorgente al layer offset
            clone_stamp_dab(
                src_data=layer.data,  # clona sullo stesso layer
                dst_data=layer.data,
                src_x=sx - layer.offset_x, src_y=sy - layer.offset_y,
                dst_x=px - layer.offset_x, dst_y=py - layer.offset_y,
                size=self._drawing_panel.brush_size,
                hardness=self._drawing_panel.hardness / 100.0,
                opacity=self._drawing_panel.opacity / 100.0,
            )
        self._clone_last_dst = (dx, dy)
        self._canvas.update_image(self._layer_stack.composite())

    def _on_clone_end(self) -> None:
        self._clone_last_dst = None
        if self._layer_stack is None: return
        composite = self._layer_stack.composite()
        self._canvas.update_image(composite)
        self._history.push(self._layer_stack.snapshot(), composite)
        self._mark_changed(); self._update_actions()
        self._refresh_history_panel()

    # ── Gradiente ─────────────────────────────────────────────────────

    def _on_gradient_applied(self, x1: float, y1: float, x2: float, y2: float) -> None:
        if self._layer_stack is None: return
        layer = self._layer_stack.active
        c1 = self._drawing_panel.current_color()
        c2 = self._drawing_panel.gradient_color2
        color1 = (c1.red(), c1.green(), c1.blue(), c1.alpha())
        color2 = (c2.red(), c2.green(), c2.blue(), c2.alpha())
        apply_gradient(
            layer.data, x1, y1, x2, y2,
            color1, color2,
            mode=self._drawing_panel.gradient_mode,
            opacity=self._drawing_panel.gradient_opacity,
            selection_mask=self._sel_mask,
            layer_offset_x=layer.offset_x,
            layer_offset_y=layer.offset_y,
        )
        self._commit()

    # ── Healing brush ─────────────────────────────────────────────────

    def _on_heal_begin(self, x: float, y: float) -> None:
        if self._layer_stack is None: return
        layer = self._layer_stack.active
        h, w = layer.data.shape[:2]
        self._heal_mask_acc = np.zeros((h, w), dtype=np.uint8)
        self._accumulate_heal_dab(x - layer.offset_x, y - layer.offset_y)

    def _on_heal_move(self, x: float, y: float) -> None:
        if self._layer_stack is None or self._heal_mask_acc is None: return
        layer = self._layer_stack.active
        self._accumulate_heal_dab(x - layer.offset_x, y - layer.offset_y)
        # Anteprima: mostra la maschera accumulata come area selezionata
        self._canvas.viewport().update()

    def _on_heal_end(self) -> None:
        if self._layer_stack is None or self._heal_mask_acc is None: return
        layer = self._layer_stack.active
        if self._heal_mask_acc.any():
            apply_healing(layer.data, self._heal_mask_acc,
                          radius=max(3, self._drawing_panel.brush_size // 4))
            # update_image invece di _commit: non chiama fit_in_view, non causa resize
            composite = self._layer_stack.composite()
            self._canvas.update_image(composite)
            self._history.push(self._layer_stack.snapshot(), composite)
            self._mark_changed()
            self._update_actions()
            self._refresh_history_panel()
        self._heal_mask_acc = None

    def _accumulate_heal_dab(self, lx: float, ly: float) -> None:
        """Aggiunge un dab circolare alla maschera heal cumulativa."""
        if self._layer_stack is None or self._heal_mask_acc is None: return
        layer = self._layer_stack.active
        dab = build_heal_mask(layer.data, lx, ly,
                              self._drawing_panel.brush_size,
                              self._drawing_panel.hardness / 100.0)
        np.maximum(self._heal_mask_acc, dab, out=self._heal_mask_acc)

    # ── Bacchetta magica ──────────────────────────────────────────────

    def _on_magic_wand_click(self, x: float, y: float) -> None:
        if self._layer_stack is None: return
        if self._wand_thread is not None: return  # elaborazione già in corso
        layer = self._layer_stack.active
        if not isinstance(layer, Layer): return

        # Coordinate canvas → layer
        lx = int(x - layer.offset_x)
        ly = int(y - layer.offset_y)
        lh, lw = layer.data.shape[:2]
        if not (0 <= lx < lw and 0 <= ly < lh): return

        dp = self._drawing_panel
        cw, ch = self._layer_stack.canvas_size

        self._wand_worker = _MagicWandWorker(
            layer.data.copy(),          # copia per thread-safety
            lx, ly,
            layer.offset_x, layer.offset_y,
            cw, ch,
            dp.wand_tolerance,
            dp.wand_contiguous,
            dp.wand_anti_alias,
        )
        self._wand_thread = QThread(self)
        self._wand_worker.moveToThread(self._wand_thread)

        self._wand_thread.started.connect(self._wand_worker.run)
        self._wand_worker.done.connect(self._on_wand_done)
        self._wand_worker.error.connect(self._on_wand_error)
        self._wand_worker.done.connect(self._wand_thread.quit)
        self._wand_worker.error.connect(self._wand_thread.quit)
        self._wand_thread.finished.connect(self._wand_worker.deleteLater)
        self._wand_thread.finished.connect(self._wand_thread.deleteLater)
        self._wand_thread.finished.connect(self._wand_cleanup)

        self.setCursor(Qt.CursorShape.WaitCursor)
        self.statusBar().showMessage("🪄 Selezione in corso…", 0)  # type: ignore[union-attr]
        self._wand_thread.start()

    def _on_wand_done(self, mask_canvas: np.ndarray) -> None:
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self._set_selection(mask_canvas)
        px = int(mask_canvas.sum())
        self.statusBar().showMessage(f"🪄 Selezionati {px} pixel", 2000)  # type: ignore[union-attr]

    def _on_wand_error(self, msg: str) -> None:
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.statusBar().showMessage(f"🪄 Errore selezione: {msg}", 3000)  # type: ignore[union-attr]

    def _wand_cleanup(self) -> None:
        self._wand_thread = None
        self._wand_worker = None

    def _cancel_wand_thread(self) -> None:
        """Interrompe il thread della bacchetta magica se attivo."""
        if self._wand_thread is not None:
            self._wand_thread.quit()
            self._wand_thread.wait(500)
            self.setCursor(Qt.CursorShape.ArrowCursor)

    # ── Forme vettoriali ──────────────────────────────────────────────

    def _on_shape_kind_changed(self, key: str) -> None:
        self._canvas.set_shape_ghost_kind(_shape_key_to_kind(key))
        if self._current_shape is not None:
            self._on_shape_placed(self._current_shape.cx, self._current_shape.cy)

    def _on_shape_placed(self, x: float, y: float) -> None:
        dp = self._drawing_panel
        kind = _shape_key_to_kind(dp.shape_kind_key)
        shape = default_shape(kind, x, y)
        _apply_panel_to_shape(dp, shape)
        self._current_shape = shape
        self._canvas.set_shape(shape)
        self.statusBar().showMessage(  # type: ignore[union-attr]
            "⬟ Forma inserita — trascina gli handle per modificarla", 2000)

    def _on_shape_params_changed(self) -> None:
        if self._current_shape is None:
            return
        _apply_panel_to_shape(self._drawing_panel, self._current_shape)
        self._canvas.refresh_shape()

    def _on_shape_handle_drag(self, hid_value: int, dx: float, dy: float) -> None:
        if self._current_shape is None:
            return
        try:
            hid = HandleID(hid_value)
        except ValueError:
            return
        self._current_shape.apply_handle_drag(hid, dx, dy)
        if hid in (HandleID.N_SIDES, HandleID.SQUIRCLE_EXP, HandleID.INNER_RADIUS):
            self._drawing_panel.sync_shape_from(self._current_shape)
        # refresh_shape: ridisegna senza resettare il drag handle
        self._canvas.refresh_shape()

    def _on_shape_apply(self) -> None:
        if self._current_shape is None or self._layer_stack is None:
            return
        layer = self._layer_stack.active
        if not isinstance(layer, Layer):
            return
        import dataclasses
        s = self._current_shape
        shape_layer = dataclasses.replace(
            s,
            cx=s.cx - layer.offset_x,
            cy=s.cy - layer.offset_y,
            arrow_ax=s.arrow_ax - layer.offset_x,
            arrow_ay=s.arrow_ay - layer.offset_y,
            arrow_bx=s.arrow_bx - layer.offset_x,
            arrow_by=s.arrow_by - layer.offset_y,
        )
        render_shape_onto(layer.data, shape_layer)
        self._current_shape = None
        self._canvas.set_shape(None)
        self._commit()
        self.statusBar().showMessage(  # type: ignore[union-attr]
            "⬟ Forma applicata sul layer.", 2000)

    # ── Upscaling ESRGAN ──────────────────────────────────────────────

    def _on_upscale_requested(self, model_name: str) -> None:
        if self._layer_stack is None:
            QMessageBox.information(self, "Upscale", "Apri prima un'immagine.")
            return
        if self._upscale_thread is not None:
            QMessageBox.information(self, "Upscale", "Upscaling già in corso.")
            return
        layer = self._layer_stack.active
        if not isinstance(layer, Layer):
            QMessageBox.information(self, "Upscale",
                "Seleziona un layer immagine (non un layer di regolazione).")
            return
        array = layer.data.copy()
        self._upscale_worker = _UpscaleWorker(array, model_name)
        self._upscale_thread = QThread(self)
        self._upscale_worker.moveToThread(self._upscale_thread)
        self._upscale_thread.started.connect(self._upscale_worker.run)
        self._upscale_worker.progress.connect(self._upscale_panel.set_status)
        self._upscale_worker.done.connect(self._on_upscale_done)
        self._upscale_worker.error.connect(self._on_upscale_error)
        self._upscale_worker.done.connect(self._upscale_thread.quit)
        self._upscale_worker.error.connect(self._upscale_thread.quit)
        self._upscale_thread.finished.connect(self._upscale_worker.deleteLater)
        self._upscale_thread.finished.connect(self._upscale_thread.deleteLater)
        self._upscale_thread.finished.connect(self._upscale_cleanup)
        self._upscale_thread.start()
        self._upscale_panel.set_busy(True)

    def _on_upscale_done(self, result: np.ndarray) -> None:
        if self._layer_stack is None: return
        self._layer_stack.set_active_data(result)
        # Aggiorna canvas_size se la dimensione è cambiata
        h, w = result.shape[:2]
        self._layer_stack._canvas_w = w
        self._layer_stack._canvas_h = h
        self._commit()
        self._upscale_panel.set_status("✓ Upscaling completato.")
        self._upscale_panel.refresh_cache_status()

    def _on_upscale_error(self, msg: str) -> None:
        QMessageBox.critical(self, "Errore upscaling", msg)
        self._upscale_panel.set_status(f"Errore: {msg}")

    def _upscale_cleanup(self) -> None:
        self._upscale_thread = None
        self._upscale_worker = None
        self._upscale_panel.set_busy(False)

    # ── Storico visuale ───────────────────────────────────────────────

    def _on_history_jump(self, index: int) -> None:
        snap = self._history.jump_to(index)
        if snap is None: return
        self._layer_stack = LayerStack.from_snapshot(snap)
        self._reset_filter_state()
        composite = self._layer_stack.composite()
        self._canvas.update_image(composite)
        self._layer_panel.refresh_from_stack(self._layer_stack)
        self._mark_changed(); self._update_layer_label(); self._update_actions()
        self._refresh_history_panel()

    def _refresh_history_panel(self) -> None:
        """Aggiorna il pannello storico — solo se visibile (risparmio prestazioni)."""
        if not self._history_dock.isVisible():
            return
        composites = []
        for c in self._history.composites:
            if c is None and self._layer_stack is not None:
                composites.append(self._layer_stack.composite())
            else:
                composites.append(c if c is not None else
                                  np.zeros((4, 4, 4), dtype=np.uint8))
        self._history_panel.set_history(composites, self._history.current_index)

    # ── Righelli e griglia ────────────────────────────────────────────

    def _toggle_rulers(self) -> None:
        visible = self._act_rulers.isChecked()
        self._canvas.set_rulers_visible(visible)

    def _toggle_grid(self) -> None:
        visible = self._act_grid.isChecked()
        self._canvas.set_grid_visible(visible)

    def _set_grid_spacing(self) -> None:
        """Dialogo per spaziatura, colore, spessore griglia + dimensione font righello."""
        from PyQt6.QtWidgets import (QDialog, QFormLayout, QDialogButtonBox,
                                     QDoubleSpinBox, QSpinBox, QColorDialog,
                                     QPushButton as _PB)
        from PyQt6.QtGui import QColor

        dlg = QDialog(self)
        dlg.setWindowTitle("Impostazioni griglia e righello")
        form = QFormLayout(dlg)

        # Spaziatura griglia
        spin_spacing = QSpinBox()
        spin_spacing.setRange(4, 2000)
        spin_spacing.setValue(self._canvas._grid_spacing)
        spin_spacing.setSuffix(" px")
        form.addRow("Spaziatura griglia:", spin_spacing)

        # Spessore linee griglia
        spin_width = QDoubleSpinBox()
        spin_width.setRange(0.5, 10.0)
        spin_width.setSingleStep(0.5)
        spin_width.setDecimals(1)
        spin_width.setValue(self._canvas._grid_width)
        spin_width.setSuffix(" px")
        form.addRow("Spessore linee:", spin_width)

        # Colore griglia
        btn_color = _PB()
        grid_color = QColor(self._canvas._grid_color)

        def _refresh_btn():
            c = grid_color
            btn_color.setStyleSheet(
                f"background:{c.name()}; border:1px solid #555; min-height:22px;")
            btn_color.setText(f"{c.name()} (α={c.alpha()})")

        _refresh_btn()

        def _pick_color():
            nonlocal grid_color
            c = QColorDialog.getColor(
                grid_color, dlg, "Colore griglia",
                QColorDialog.ColorDialogOption.ShowAlphaChannel)
            if c.isValid():
                grid_color = c
                _refresh_btn()

        btn_color.clicked.connect(_pick_color)
        form.addRow("Colore griglia:", btn_color)

        # Font righello
        spin_font = QSpinBox()
        spin_font.setRange(6, 20)
        spin_font.setValue(self._canvas._ruler_font_pt)
        spin_font.setSuffix(" pt")
        form.addRow("Font righello:", spin_font)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        form.addRow(buttons)

        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._canvas.set_grid_spacing(spin_spacing.value())
            self._canvas.set_grid_style(grid_color, spin_width.value())
            self._canvas.set_ruler_font_size(spin_font.value())

    def _apply_clone_dab(self, dx: float, dy: float) -> None:
        if self._layer_stack is None or self._clone_source_canvas is None: return
        src_cx, src_cy = self._clone_source_canvas
        layer = self._layer_stack.active
        clone_stamp_dab(
            src_data=layer.data, dst_data=layer.data,
            src_x=src_cx - layer.offset_x, src_y=src_cy - layer.offset_y,
            dst_x=dx - layer.offset_x,     dst_y=dy - layer.offset_y,
            size=self._drawing_panel.brush_size,
            hardness=self._drawing_panel.hardness / 100.0,
            opacity=self._drawing_panel.opacity / 100.0,
        )
        self._canvas.update_image(self._layer_stack.composite())

    # ── Slot Layer di regolazione ────────────────────────────────────

    def _on_adj_add(self, kind: str, params: dict) -> None:
        if self._layer_stack is None: return
        kind_names = {
            "levels": "Livelli", "hsl": "HSL",
            "color_balance": "Bilanciamento colore",
            "curves": "Curve", "exposure": "Esposizione",
        }
        name = kind_names.get(kind, kind.capitalize())
        adj = AdjustmentLayer(name=name, kind=kind, params=params)
        # Inserisce sopra il layer attivo
        parent_list = self._layer_stack.get_parent_list(self._layer_stack.active_path)
        insert_at = self._layer_stack.active_path[-1] + 1
        parent_list.insert(insert_at, adj)
        self._commit()
        self.statusBar().showMessage(f"Layer di regolazione «{name}» aggiunto.", 2500)  # type: ignore[union-attr]

    def _on_adj_update(self, params: dict) -> None:
        """Aggiorna i parametri del nodo selezionato se è un AdjustmentLayer."""
        if self._layer_stack is None: return
        try:
            node = self._layer_stack.get_node(self._layer_stack.active_path)
        except Exception:
            return
        if not isinstance(node, AdjustmentLayer): return
        import copy
        node.params = copy.deepcopy(params)
        # Aggiorna il canvas senza push history (è solo una preview)
        self._canvas.update_image(self._layer_stack.composite())
        # Push history
        self._history.push(self._layer_stack.snapshot())
        self._mark_changed()

    # ── Slot Trasformazione libera ────────────────────────────────────

    def _on_ft_preview(self, params: dict) -> None:
        if self._layer_stack is None: return
        # Salva i dati originali la prima volta
        if self._ft_original_data is None:
            self._ft_original_data = self._layer_stack.active.data.copy()
        data = self._ft_original_data
        try:
            preview = apply_transform(
                data,
                scale_x=params["scale_x"], scale_y=params["scale_y"],
                angle_deg=params["angle_deg"],
                skew_x=params["skew_x"], skew_y=params["skew_y"],
                expand=params.get("expand", True),
            )
        except Exception:
            return
        # Mostra anteprima senza committare
        tmp_stack = self._layer_stack.composite_with_override(
            self._layer_stack.active_path, preview)
        self._canvas.update_image(tmp_stack)

    def _on_ft_apply(self, params: dict) -> None:
        if self._layer_stack is None: return
        data = self._ft_original_data if self._ft_original_data is not None                else self._layer_stack.active.data
        try:
            result = apply_transform(
                data,
                scale_x=params["scale_x"], scale_y=params["scale_y"],
                angle_deg=params["angle_deg"],
                skew_x=params["skew_x"], skew_y=params["skew_y"],
                expand=params.get("expand", True),
            )
        except Exception as exc:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Errore trasformazione", str(exc)); return
        self._layer_stack.set_active_data(result)
        self._ft_original_data = None
        self._commit()
        self.statusBar().showMessage("Trasformazione applicata.", 2000)  # type: ignore[union-attr]

    def _on_ft_reset(self) -> None:
        if self._layer_stack is None: return
        if self._ft_original_data is not None:
            self._layer_stack.set_active_data(self._ft_original_data)
            self._ft_original_data = None
            self._canvas.update_image(self._layer_stack.composite())

    def _on_canvas_cleared(self) -> None:
        self._lbl_pixel.setText("")

    def _on_about(self) -> None:
        QMessageBox.about(self, "Info",
            "<b>Image Suite 0.8.0</b><br><br>"
            "Python 3.12 · PyQt6 · Pillow · OpenCV<br><br>"
            "Fase 8 — Layer groups · Albero gerarchico")

    # ── Commit ────────────────────────────────────────────────────────

    def _commit(self) -> None:
        if self._layer_stack is None: return
        composite = self._layer_stack.composite()
        self._history.push(self._layer_stack.snapshot(), composite)
        self._canvas.set_image(composite)
        self._layer_panel.refresh_from_stack(self._layer_stack)
        self._mark_changed(); self._update_actions(); self._update_layer_label()
        h, w = composite.shape[:2]
        self._lbl_size.setText(f"{w} × {h} px")
        self._refresh_history_panel()

    # ── Helpers ───────────────────────────────────────────────────────

    def _reset_filter_state(self) -> None:
        self._filters = FilterStack(); self._preview = None
        self._filter_panel._reset()

    def _check_unsaved(self) -> bool:
        if not self._unsaved_changes: return True
        reply = QMessageBox.question(self, "Modifiche non salvate",
            "Vuoi continuare senza salvare?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        return reply == QMessageBox.StandardButton.Yes

    def _mark_changed(self) -> None:
        self._unsaved_changes = True; self._update_title()

    def _update_title(self) -> None:
        if self._project_path:
            name = self._project_path.name
        elif self._current_path:
            name = self._current_path.name
        else:
            name = "Senza titolo"
        marker = " •" if self._unsaved_changes else ""
        self.setWindowTitle(f"{name}{marker} — Image Suite")
        display = str(self._project_path or self._current_path or name)
        self._lbl_file.setText(display)

    def _update_size_label(self, array: np.ndarray) -> None:
        h, w = array.shape[:2]; self._lbl_size.setText(f"{w} × {h} px")

    def _update_layer_label(self) -> None:
        if self._layer_stack is None: self._lbl_layers.setText(""); return
        ls = self._layer_stack
        try:
            active = ls.active
            path = ls.active_path
            depth = len(path) - 1
            has_mask = " 🎭" if active.mask is not None else ""
            off = (f" [{active.offset_x:+d},{active.offset_y:+d}]"
                   if (active.offset_x or active.offset_y) else "")
            indent = "  " * depth + ("↳ " if depth else "")
            self._lbl_layers.setText(f"{indent}{active.name}{has_mask}{off}")
        except Exception:
            self._lbl_layers.setText("")

    def _update_actions(self) -> None:
        has = self._layer_stack is not None
        self._act_save.setEnabled(has); self._act_save_as.setEnabled(has)
        self._act_close.setEnabled(has); self._act_import.setEnabled(has)
        self._act_flatten.setEnabled(has)
        self._act_proj_save.setEnabled(has)
        self._act_proj_saveas.setEnabled(has)
        self._act_ora_save.setEnabled(has)
        self._act_copy.setEnabled(has)
        self._act_paste.setEnabled(True)   # incolla sempre disponibile
        self._act_undo.setEnabled(self._history.can_undo)
        self._act_redo.setEnabled(self._history.can_redo)

    def _action(self, text: str, shortcut, slot) -> QAction:
        a = QAction(text, self)
        if shortcut is not None: a.setShortcut(shortcut)
        a.triggered.connect(slot)
        return a

    def closeEvent(self, event) -> None:
        if self._unsaved_changes:
            reply = QMessageBox.question(self, "Uscire?", "Modifiche non salvate. Uscire?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply != QMessageBox.StandardButton.Yes:
                event.ignore(); return
        event.accept()
