"""
ui/shape_panel.py
-----------------
Pannello di controllo per lo strumento forme.
Espone:
  • Selettore tipo forma
  • Colore riempimento / bordo / opacità / spessore bordo
  • Parametri specifici (lati poligono, raggi stella, esponente squircle, …)
  • Pulsante "Carica SVG"
  • Pulsante "Applica sul layer"

Segnali
-------
  shape_changed()         — qualsiasi parametro di stile è cambiato
  shape_apply_requested() — l'utente vuole stampare la forma sul layer
  shape_kind_changed(str) — il tipo di forma è cambiato
  svg_load_requested()    — l'utente vuole caricare un SVG
"""
from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QSlider, QLabel, QColorDialog,
    QSpinBox, QDoubleSpinBox, QFrame, QComboBox,
    QFileDialog, QCheckBox,
)

from core.shapes import ShapeKind, ShapeStyle, Shape


# Mappa nome UI → ShapeKind
_KIND_MAP: list[tuple[str, ShapeKind]] = [
    ("Ellisse / Cerchio",  ShapeKind.ELLIPSE),
    ("Rettangolo",         ShapeKind.RECT),
    ("Poligono regolare",  ShapeKind.POLYGON),
    ("Stella",             ShapeKind.STAR),
    ("Squircle",           ShapeKind.SQUIRCLE),
    ("Freccia",            ShapeKind.ARROW),
    ("Logo SVG",           ShapeKind.SVG),
]


class _ColorButton(QPushButton):
    """Pulsante che mostra un colore e apre un color picker al click."""
    color_changed = pyqtSignal(QColor)

    def __init__(self, color: QColor = QColor(100, 149, 237, 200), parent=None):
        super().__init__(parent)
        self._color = QColor(color)
        self.setFixedSize(36, 24)
        self._refresh()
        self.clicked.connect(self._pick)

    def _refresh(self):
        c = self._color
        self.setStyleSheet(
            f"background:rgba({c.red()},{c.green()},{c.blue()},{c.alpha()});"
            "border:1px solid #666; border-radius:3px;")

    def color(self) -> QColor:
        return QColor(self._color)

    def set_color(self, c: QColor):
        self._color = QColor(c); self._refresh()

    def _pick(self):
        c = QColorDialog.getColor(
            self._color, self, "Scegli colore",
            QColorDialog.ColorDialogOption.ShowAlphaChannel)
        if c.isValid():
            self._color = c; self._refresh()
            self.color_changed.emit(c)


class ShapePanel(QWidget):
    shape_changed         = pyqtSignal()
    shape_apply_requested = pyqtSignal()
    shape_kind_changed    = pyqtSignal(str)   # nome stringa del kind
    svg_load_requested    = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setSpacing(6)
        root.setContentsMargins(4, 4, 4, 4)

        # ── Tipo forma ───────────────────────────────────────────────
        kind_box = QGroupBox("Forma")
        kl = QVBoxLayout(kind_box)
        self._kind_combo = QComboBox()
        for label, _ in _KIND_MAP:
            self._kind_combo.addItem(label)
        self._kind_combo.currentIndexChanged.connect(self._on_kind_changed)
        kl.addWidget(self._kind_combo)

        # Pulsante carica SVG
        self._btn_svg = QPushButton("📂 Carica SVG…")
        self._btn_svg.setVisible(False)
        self._btn_svg.clicked.connect(self.svg_load_requested)
        kl.addWidget(self._btn_svg)

        self._lbl_svg_file = QLabel("Nessun file caricato")
        self._lbl_svg_file.setStyleSheet("color:#888; font-size:10px;")
        self._lbl_svg_file.setVisible(False)
        kl.addWidget(self._lbl_svg_file)
        root.addWidget(kind_box)

        # ── Stile ────────────────────────────────────────────────────
        style_box = QGroupBox("Stile")
        sl = QVBoxLayout(style_box)

        # Fill
        fill_row = QHBoxLayout()
        fill_row.addWidget(QLabel("Riempimento:"))
        self._fill_btn = _ColorButton(QColor(100, 149, 237, 200))
        self._fill_btn.color_changed.connect(self._emit_changed)
        fill_row.addWidget(self._fill_btn)
        self._chk_fill = QCheckBox("Visibile")
        self._chk_fill.setChecked(True)
        self._chk_fill.toggled.connect(self._emit_changed)
        fill_row.addWidget(self._chk_fill)
        fill_row.addStretch()
        sl.addLayout(fill_row)

        # Stroke
        stroke_row = QHBoxLayout()
        stroke_row.addWidget(QLabel("Bordo:"))
        self._stroke_btn = _ColorButton(QColor(30, 30, 30, 255))
        self._stroke_btn.color_changed.connect(self._emit_changed)
        stroke_row.addWidget(self._stroke_btn)
        self._stroke_w = QSpinBox()
        self._stroke_w.setRange(0, 60)
        self._stroke_w.setValue(2)
        self._stroke_w.setSuffix(" px")
        self._stroke_w.setFixedWidth(70)
        self._stroke_w.valueChanged.connect(self._emit_changed)
        stroke_row.addWidget(self._stroke_w)
        stroke_row.addStretch()
        sl.addLayout(stroke_row)

        # Opacità globale
        op_row = QHBoxLayout()
        op_row.addWidget(QLabel("Opacità:"))
        self._opacity_sl = QSlider(Qt.Orientation.Horizontal)
        self._opacity_sl.setRange(0, 100)
        self._opacity_sl.setValue(100)
        self._lbl_opacity = QLabel("100%")
        self._lbl_opacity.setFixedWidth(38)
        self._opacity_sl.valueChanged.connect(
            lambda v: (self._lbl_opacity.setText(f"{v}%"), self._emit_changed()))
        op_row.addWidget(self._opacity_sl, 1)
        op_row.addWidget(self._lbl_opacity)
        sl.addLayout(op_row)
        root.addWidget(style_box)

        # ── Parametri specifici per tipo ─────────────────────────────
        self._params_box = QGroupBox("Parametri")
        pl = QVBoxLayout(self._params_box)

        # Angoli arrotondati (rect)
        self._corner_row = QHBoxLayout()
        self._corner_row.addWidget(QLabel("Arrotondamento:"))
        self._corner_r = QSpinBox()
        self._corner_r.setRange(0, 200)
        self._corner_r.setValue(0)
        self._corner_r.setSuffix(" px")
        self._corner_r.valueChanged.connect(self._emit_changed)
        self._corner_row.addWidget(self._corner_r)
        self._corner_row.addStretch()
        pl.addLayout(self._corner_row)

        # Numero lati (polygon / star)
        self._sides_row = QHBoxLayout()
        self._sides_row.addWidget(QLabel("Lati / punte:"))
        self._n_sides = QSpinBox()
        self._n_sides.setRange(3, 20)
        self._n_sides.setValue(6)
        self._n_sides.valueChanged.connect(self._emit_changed)
        self._sides_row.addWidget(self._n_sides)
        self._sides_row.addStretch()
        pl.addLayout(self._sides_row)

        # Raggio interno (star)
        self._inner_row = QHBoxLayout()
        self._inner_row.addWidget(QLabel("Raggio interno:"))
        self._inner_ratio = QDoubleSpinBox()
        self._inner_ratio.setRange(0.05, 0.95)
        self._inner_ratio.setSingleStep(0.05)
        self._inner_ratio.setValue(0.45)
        self._inner_ratio.valueChanged.connect(self._emit_changed)
        self._inner_row.addWidget(self._inner_ratio)
        self._inner_row.addStretch()
        pl.addLayout(self._inner_row)

        # Esponente squircle
        self._sq_row = QHBoxLayout()
        self._sq_row.addWidget(QLabel("Esponente:"))
        self._sq_exp = QDoubleSpinBox()
        self._sq_exp.setRange(2.0, 20.0)
        self._sq_exp.setSingleStep(0.5)
        self._sq_exp.setValue(4.0)
        self._sq_exp.valueChanged.connect(self._emit_changed)
        self._sq_row.addWidget(self._sq_exp)
        self._sq_row.addStretch()
        pl.addLayout(self._sq_row)

        # Freccia: larghezza gambo
        self._arrow_row = QHBoxLayout()
        self._arrow_row.addWidget(QLabel("Larghezza gambo:"))
        self._arrow_w = QSpinBox()
        self._arrow_w.setRange(4, 200)
        self._arrow_w.setValue(22)
        self._arrow_w.setSuffix(" px")
        self._arrow_w.valueChanged.connect(self._emit_changed)
        self._arrow_row.addWidget(self._arrow_w)
        self._arrow_row.addStretch()
        pl.addLayout(self._arrow_row)

        # Freccia: proporzione testa
        self._arrowh_row = QHBoxLayout()
        self._arrowh_row.addWidget(QLabel("Proporzione testa:"))
        self._arrow_head = QDoubleSpinBox()
        self._arrow_head.setRange(0.1, 0.9)
        self._arrow_head.setSingleStep(0.05)
        self._arrow_head.setValue(0.30)
        self._arrow_head.valueChanged.connect(self._emit_changed)
        self._arrowh_row.addWidget(self._arrow_head)
        self._arrowh_row.addStretch()
        pl.addLayout(self._arrowh_row)

        root.addWidget(self._params_box)

        # ── Hint interazione ─────────────────────────────────────────
        self._hint = QLabel(
            "<i>Clicca sul canvas per inserire la forma.<br>"
            "Trascina gli handle per modificarla.</i>")
        self._hint.setWordWrap(True)
        self._hint.setStyleSheet("color:#888; font-size:10px;")
        root.addWidget(self._hint)

        # ── Applica ──────────────────────────────────────────────────
        btn_apply = QPushButton("✓ Applica forma sul layer")
        btn_apply.setStyleSheet("font-weight:bold; padding:5px;")
        btn_apply.clicked.connect(self.shape_apply_requested)
        root.addWidget(btn_apply)
        root.addStretch()

        # Init visibilità
        self._on_kind_changed(0)

    # ── Internal ─────────────────────────────────────────────────────

    def _emit_changed(self, *_):
        self.shape_changed.emit()

    def _on_kind_changed(self, idx: int):
        kind = _KIND_MAP[idx][1]
        is_rect     = kind == ShapeKind.RECT
        is_poly     = kind == ShapeKind.POLYGON
        is_star     = kind == ShapeKind.STAR
        is_sq       = kind == ShapeKind.SQUIRCLE
        is_arrow    = kind == ShapeKind.ARROW
        is_svg      = kind == ShapeKind.SVG
        is_poly_star = is_poly or is_star

        self._show_row(self._corner_row,  is_rect)
        self._show_row(self._sides_row,   is_poly_star)
        self._show_row(self._inner_row,   is_star)
        self._show_row(self._sq_row,      is_sq)
        self._show_row(self._arrow_row,   is_arrow)
        self._show_row(self._arrowh_row,  is_arrow)

        self._btn_svg.setVisible(is_svg)
        self._lbl_svg_file.setVisible(is_svg)
        self._params_box.setVisible(
            is_rect or is_poly_star or is_sq or is_arrow)
        self._chk_fill.setVisible(not is_svg)

        self.shape_kind_changed.emit(_KIND_MAP[idx][0])
        self._emit_changed()

    @staticmethod
    def _show_row(layout: QHBoxLayout, visible: bool):
        for i in range(layout.count()):
            w = layout.itemAt(i).widget()
            if w:
                w.setVisible(visible)

    def set_svg_filename(self, name: str):
        self._lbl_svg_file.setText(name)

    # ── Accessors ────────────────────────────────────────────────────

    @property
    def current_kind(self) -> ShapeKind:
        return _KIND_MAP[self._kind_combo.currentIndex()][1]

    def build_style(self) -> ShapeStyle:
        fc = self._fill_btn.color()
        sc = self._stroke_btn.color()
        fill_a = fc.alpha() if self._chk_fill.isChecked() else 0
        return ShapeStyle(
            fill_color   = (fc.red(), fc.green(), fc.blue(), fill_a),
            stroke_color = (sc.red(), sc.green(), sc.blue(), sc.alpha()),
            stroke_width = self._stroke_w.value(),
            opacity      = self._opacity_sl.value() / 100.0,
        )

    def apply_params_to_shape(self, shape: Shape):
        """Copia i parametri specifici del pannello sulla shape."""
        shape.style        = self.build_style()
        shape.n_sides      = self._n_sides.value()
        shape.inner_ratio  = self._inner_ratio.value()
        shape.sq_exp       = self._sq_exp.value()
        shape.corner_r     = float(self._corner_r.value())
        shape.arrow_width  = float(self._arrow_w.value())
        shape.arrow_head_ratio = self._arrow_head.value()

    def sync_from_shape(self, shape: Shape):
        """Aggiorna il pannello leggendo i valori dalla shape."""
        self._opacity_sl.blockSignals(True)
        self._opacity_sl.setValue(int(shape.style.opacity * 100))
        self._opacity_sl.blockSignals(False)

        fc = shape.style.fill_color
        self._fill_btn.set_color(QColor(*fc))
        sc = shape.style.stroke_color
        self._stroke_btn.set_color(QColor(*sc))

        self._stroke_w.blockSignals(True)
        self._stroke_w.setValue(shape.style.stroke_width)
        self._stroke_w.blockSignals(False)

        self._n_sides.blockSignals(True)
        self._n_sides.setValue(shape.n_sides)
        self._n_sides.blockSignals(False)

        self._inner_ratio.blockSignals(True)
        self._inner_ratio.setValue(shape.inner_ratio)
        self._inner_ratio.blockSignals(False)

        self._sq_exp.blockSignals(True)
        self._sq_exp.setValue(shape.sq_exp)
        self._sq_exp.blockSignals(False)
