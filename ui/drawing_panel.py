"""
ui/drawing_panel.py
-------------------
Pannello strumenti di disegno: pennello, gomma, sfumino, contagocce, testo.
"""
from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QFontDatabase
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QSlider, QLabel, QColorDialog,
    QGroupBox, QFrame, QButtonGroup, QLineEdit,
    QFontComboBox, QSpinBox, QDoubleSpinBox, QCheckBox, QScrollArea, QComboBox,
    QPlainTextEdit, QFileDialog,
)

TOOLS = [
    ("brush",          "🖌",    "Pennello  [B]"),
    ("eraser",         "⬜",    "Gomma     [E]"),
    ("smudge",         "💧",    "Sfumino   [U]"),
    ("eyedropper",     "🔍",    "Contagocce [I]"),
    ("text",           "T",     "Testo     [T]"),
    ("select_rect",    "⬜Sel", "Selezione rettangolare [S]"),
    ("select_ellipse", "◯Sel", "Selezione ellittica"),
    ("select_lasso",   "🔗",    "Lasso libero [L]"),
    ("clone",          "🔁",    "Timbro clone [C]"),
    ("gradient",       "🌈",    "Gradiente [G]"),
    ("heal",           "🩹",    "Guarigione [H]"),
    ("magic_wand",     "🪄",    "Bacchetta magica [W]"),
    ("shape",          "⬟",    "Forme vettoriali [F]"),
]


class _ColorSwatch(QFrame):
    color_changed = pyqtSignal(QColor)

    def __init__(self, color: QColor = QColor(0, 0, 0, 255), parent=None):
        super().__init__(parent)
        self._color = color
        self.setFixedSize(40, 40)
        self.setFrameShape(QFrame.Shape.Box)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._refresh()

    def _refresh(self):
        c = self._color
        self.setStyleSheet(
            f"background-color: rgba({c.red()},{c.green()},{c.blue()},{c.alpha()});"
            "border: 2px solid #666;")

    def color(self) -> QColor: return QColor(self._color)

    def set_color(self, c: QColor) -> None:
        self._color = QColor(c); self._refresh()

    def mousePressEvent(self, _event):
        dlg = QColorDialog(self._color, self)
        dlg.setWindowTitle("Scegli colore")
        dlg.setOption(QColorDialog.ColorDialogOption.ShowAlphaChannel, True)
        # Forza colori UI stabili indipendentemente dal tema di sistema
        dlg.setStyleSheet(
            "QColorDialog, QColorDialog * { background: #2b2b2b; color: #dddddd; }"
            "QPushButton { background: #444; color: #ddd; border: 1px solid #666; "
            "padding: 4px 10px; border-radius: 3px; }"
            "QPushButton:hover { background: #555; }"
            "QPushButton:pressed { background: #333; }"
        )
        if dlg.exec():
            picked = dlg.currentColor()
            if picked.isValid():
                self._color = picked; self._refresh()
                self.color_changed.emit(picked)


class DrawingPanel(QWidget):
    tool_changed        = pyqtSignal(str)
    color_changed       = pyqtSignal(QColor)
    size_changed        = pyqtSignal(int)
    hardness_changed    = pyqtSignal(int)
    opacity_changed     = pyqtSignal(int)
    text_apply          = pyqtSignal()
    text_params_changed = pyqtSignal()
    # Selezione
    sel_copy_requested  = pyqtSignal()
    sel_cut_requested   = pyqtSignal()
    sel_fill_requested  = pyqtSignal()
    sel_invert_requested = pyqtSignal()
    sel_deselect_requested = pyqtSignal()
    sel_feather_changed = pyqtSignal(int)   # raggio feather
    # Forme
    shape_changed       = pyqtSignal()
    shape_apply         = pyqtSignal()
    shape_kind_changed  = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._active_tool: str = ""

        # Scroll area esterna (il pannello può essere alto)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        outer.addWidget(scroll)

        content = QWidget()
        root = QVBoxLayout(content)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(8)
        scroll.setWidget(content)

        # ── Strumenti (griglia 3+2) ────────────────────────────────────
        tool_box = QGroupBox("Strumento")
        grid = QGridLayout(tool_box)
        grid.setSpacing(4)
        self._btn_group = QButtonGroup(self)
        self._btn_group.setExclusive(False)
        self._btns: dict[str, QPushButton] = {}

        for i, (key, icon, tip) in enumerate(TOOLS):
            btn = QPushButton(icon)
            btn.setToolTip(tip)
            btn.setCheckable(True)
            btn.setFixedSize(50, 42)
            btn.setFont(QFont("Segoe UI Emoji", 15))
            self._btn_group.addButton(btn)
            self._btns[key] = btn
            btn.clicked.connect(lambda checked, k=key: self._on_btn_clicked(k, checked))
            grid.addWidget(btn, i // 3, i % 3)

        root.addWidget(tool_box)

        # ── Colore ────────────────────────────────────────────────────
        color_box = QGroupBox("Colore")
        color_row = QHBoxLayout(color_box)
        self._swatch = _ColorSwatch(QColor(0, 0, 0, 255))
        self._lbl_rgba = QLabel("R:0 G:0 B:0 A:255")
        self._lbl_rgba.setStyleSheet("font-size: 10px; color: #aaa;")
        self._swatch.color_changed.connect(self._on_color_changed)
        color_row.addWidget(self._swatch)
        color_row.addWidget(self._lbl_rgba, 1)
        root.addWidget(color_box)

        # ── Dimensione ────────────────────────────────────────────────
        self._size_box = QGroupBox("Dimensione")
        self._size_slider, self._lbl_size = self._make_slider_row(
            self._size_box, 1, 300, 20, lambda v: f"{v} px")
        self._size_slider.valueChanged.connect(self.size_changed)
        root.addWidget(self._size_box)

        # ── Tipo pennello ─────────────────────────────────────────────
        self._brush_type_box = QGroupBox("Tipo pennello")
        bt_layout = QVBoxLayout(self._brush_type_box)
        self._brush_type_combo = QComboBox()
        self._brush_type_combo.addItems([
            "Pennello standard",
            "Aerografo",
            "Matita",
            "Calligrafia",
        ])
        self._brush_type_combo.currentIndexChanged.connect(self._on_brush_type_changed)
        bt_layout.addWidget(self._brush_type_combo)

        # Punta calligrafica (angolo + larghezza relativa)
        self._calli_angle_row = QHBoxLayout()
        self._calli_angle_row.addWidget(QLabel("Angolo:"))
        self._calli_angle = QSpinBox()
        self._calli_angle.setRange(0, 179); self._calli_angle.setValue(45)
        self._calli_angle.setSuffix("°")
        self._calli_angle_row.addWidget(self._calli_angle)
        self._calli_angle_row.addStretch()
        bt_layout.addLayout(self._calli_angle_row)

        self._calli_ratio_row = QHBoxLayout()
        self._calli_ratio_row.addWidget(QLabel("Larghezza punta:"))
        self._calli_ratio = QSpinBox()
        self._calli_ratio.setRange(5, 95); self._calli_ratio.setValue(25)
        self._calli_ratio.setSuffix("%")
        self._calli_ratio_row.addWidget(self._calli_ratio)
        self._calli_ratio_row.addStretch()
        bt_layout.addLayout(self._calli_ratio_row)

        self._brush_type_box.setVisible(False)  # visibile solo con brush/eraser
        root.addWidget(self._brush_type_box)
        self._on_brush_type_changed(0)  # init visibilità punta

        # ── Durezza / Intensità ────────────────────────────────────────
        self._hard_box = QGroupBox("Durezza")
        self._hard_slider, self._lbl_hard = self._make_slider_row(
            self._hard_box, 0, 100, 80, lambda v: f"{v}%")
        self._hard_slider.valueChanged.connect(self.hardness_changed)
        root.addWidget(self._hard_box)

        # ── Opacità ───────────────────────────────────────────────────
        self._op_box = QGroupBox("Opacità")
        self._op_slider, self._lbl_op = self._make_slider_row(
            self._op_box, 1, 100, 100, lambda v: f"{v}%")
        self._op_slider.valueChanged.connect(self.opacity_changed)
        root.addWidget(self._op_box)

        # ── Sezione Testo (visibile solo con tool "text") ─────────────
        self._text_box = QGroupBox("Testo")
        text_layout = QVBoxLayout(self._text_box)

        # Contenuto — QPlainTextEdit per supportare il tasto Invio / a capo
        text_layout.addWidget(QLabel("Contenuto:"))
        self._txt_input = QPlainTextEdit()
        self._txt_input.setPlaceholderText("Scrivi qui il testo…\n(Invio = a capo)")
        self._txt_input.setFixedHeight(72)
        self._txt_input.textChanged.connect(self._on_text_params_changed)
        text_layout.addWidget(self._txt_input)

        # Font family
        self._font_combo = QFontComboBox()
        self._font_combo.setCurrentFont(QFont("Arial"))
        self._font_combo.currentFontChanged.connect(self._on_text_params_changed)
        text_layout.addWidget(QLabel("Font:"))
        text_layout.addWidget(self._font_combo)

        # Dimensione + Bold + Italic
        style_row = QHBoxLayout()
        style_row.addWidget(QLabel("Dim:"))
        self._font_size = QSpinBox()
        self._font_size.setRange(6, 400)
        self._font_size.setValue(36)
        self._font_size.setSuffix(" px")
        self._font_size.valueChanged.connect(self._on_text_params_changed)
        style_row.addWidget(self._font_size)
        self._chk_bold   = QCheckBox("G")
        self._chk_italic = QCheckBox("I")
        self._chk_bold.setFont(QFont("Arial", 9, QFont.Weight.Bold))
        self._chk_italic.setFont(QFont("Arial", 9, -1, True))
        self._chk_bold.toggled.connect(self._on_text_params_changed)
        self._chk_italic.toggled.connect(self._on_text_params_changed)
        style_row.addWidget(self._chk_bold)
        style_row.addWidget(self._chk_italic)
        style_row.addStretch()
        text_layout.addLayout(style_row)

        # Opacità testo
        op_row = QHBoxLayout()
        op_row.addWidget(QLabel("Opacità:"))
        self._txt_opacity = QSlider(Qt.Orientation.Horizontal)
        self._txt_opacity.setRange(0, 100)
        self._txt_opacity.setValue(100)
        self._lbl_txt_opacity = QLabel("100%")
        self._lbl_txt_opacity.setFixedWidth(36)
        self._txt_opacity.valueChanged.connect(
            lambda v: (self._lbl_txt_opacity.setText(f"{v}%"),
                       self._on_text_params_changed()))
        op_row.addWidget(self._txt_opacity, 1)
        op_row.addWidget(self._lbl_txt_opacity)
        text_layout.addLayout(op_row)

        # Effetto WordArt
        text_layout.addWidget(QLabel("Effetto:"))
        self._txt_effect = QComboBox()
        self._txt_effect.addItems([
            "Nessuno",
            "Ombra",
            "Bordo",
            "Gradiente",
            "Arco",
            "Onda",
        ])
        self._txt_effect.currentIndexChanged.connect(self._on_text_effect_changed)
        text_layout.addWidget(self._txt_effect)

        # Colore secondario (ombra / bordo / gradiente)
        self._effect_color_row = QHBoxLayout()
        self._lbl_effect_color = QLabel("Colore effetto:")
        self._btn_effect_color = QPushButton()
        self._btn_effect_color.setFixedSize(32, 22)
        self._effect_color_val = QColor(0, 0, 0, 200)
        self._btn_effect_color.setStyleSheet(
            f"background:{self._effect_color_val.name()}; border:1px solid #666;")
        self._btn_effect_color.clicked.connect(self._pick_effect_color)
        self._effect_color_row.addWidget(self._lbl_effect_color)
        self._effect_color_row.addWidget(self._btn_effect_color)
        self._effect_color_row.addStretch()
        text_layout.addLayout(self._effect_color_row)

        # Spessore bordo (visibile solo con effetto "Bordo")
        self._outline_row = QHBoxLayout()
        self._outline_row.addWidget(QLabel("Spessore bordo:"))
        self._txt_outline_w = QSpinBox()
        self._txt_outline_w.setRange(1, 30)
        self._txt_outline_w.setValue(2)
        self._txt_outline_w.valueChanged.connect(self._on_text_params_changed)
        self._outline_row.addWidget(self._txt_outline_w)
        self._outline_row.addStretch()
        text_layout.addLayout(self._outline_row)

        self._on_text_effect_changed(0)   # inizia con "Nessuno"

        # Rotazione testo
        rot_row = QHBoxLayout()
        rot_row.addWidget(QLabel("Rotazione:"))
        self._txt_rotation = QSpinBox()
        self._txt_rotation.setRange(-180, 180)
        self._txt_rotation.setValue(0)
        self._txt_rotation.setSuffix("°")
        self._txt_rotation.setWrapping(True)
        self._txt_rotation.valueChanged.connect(self._on_text_params_changed)
        rot_row.addWidget(self._txt_rotation)
        rot_row.addStretch()
        text_layout.addLayout(rot_row)

        # Istruzioni
        hint = QLabel("<i>Clicca sul canvas per posizionare.<br>"
                      "Trascina per spostare.<br>"
                      "Handle ↻ per ruotare.</i>")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #888; font-size: 10px;")
        text_layout.addWidget(hint)

        # Applica
        btn_apply_text = QPushButton("✓ Applica testo sul layer")
        btn_apply_text.setStyleSheet("font-weight: bold; padding: 5px;")
        btn_apply_text.clicked.connect(self.text_apply.emit)
        text_layout.addWidget(btn_apply_text)

        self._text_box.setVisible(False)
        root.addWidget(self._text_box)

        # ── Sezione Selezione ─────────────────────────────────────────
        self._sel_box = QGroupBox("Selezione")
        sel_layout = QVBoxLayout(self._sel_box)

        feather_row = QHBoxLayout()
        feather_row.addWidget(QLabel("Feather:"))
        self._spin_feather = QSpinBox()
        self._spin_feather.setRange(0, 100); self._spin_feather.setValue(0)
        self._spin_feather.setSuffix(" px")
        self._spin_feather.valueChanged.connect(self.sel_feather_changed)
        feather_row.addWidget(self._spin_feather)
        sel_layout.addLayout(feather_row)

        btn_sel_row1 = QHBoxLayout()
        for label, signal in [("Copia", self.sel_copy_requested),
                               ("Taglia", self.sel_cut_requested),
                               ("Riempi", self.sel_fill_requested)]:
            btn = QPushButton(label)
            btn.clicked.connect(signal.emit)
            btn_sel_row1.addWidget(btn)
        sel_layout.addLayout(btn_sel_row1)

        btn_sel_row2 = QHBoxLayout()
        for label, signal in [("Inverti", self.sel_invert_requested),
                               ("Deseleziona", self.sel_deselect_requested)]:
            btn = QPushButton(label)
            btn.clicked.connect(signal.emit)
            btn_sel_row2.addWidget(btn)
        sel_layout.addLayout(btn_sel_row2)

        self._sel_box.setVisible(False)
        root.addWidget(self._sel_box)

        # ── Suggerimento Clone ────────────────────────────────────────
        self._clone_hint = QLabel(
            "<b>Alt+click</b> sull'immagine per impostare il punto sorgente.<br>"
            "Poi trascina per clonare."
        )
        self._clone_hint.setWordWrap(True)
        self._clone_hint.setStyleSheet("color: #aaa; font-size: 10px;")
        self._clone_hint.setVisible(False)
        root.addWidget(self._clone_hint)

        # ── Sezione Gradiente ─────────────────────────────────────────
        self._grad_box = QGroupBox("Gradiente")
        grad_layout = QVBoxLayout(self._grad_box)
        # Colore 2
        grad_c2_row = QHBoxLayout()
        grad_c2_row.addWidget(QLabel("Colore 2:"))
        self._swatch2 = _ColorSwatch(QColor(255, 255, 255, 255))
        self._swatch2.color_changed.connect(lambda c: None)
        grad_c2_row.addWidget(self._swatch2)
        grad_c2_row.addStretch()
        grad_layout.addLayout(grad_c2_row)
        # Tipo gradiente
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Tipo:"))
        self._combo_grad_mode = QComboBox()
        self._combo_grad_mode.addItem("Lineare", "linear")
        self._combo_grad_mode.addItem("Radiale", "radial")
        mode_row.addWidget(self._combo_grad_mode)
        grad_layout.addLayout(mode_row)
        # Opacità gradiente
        op_row = QHBoxLayout()
        op_row.addWidget(QLabel("Opacità:"))
        self._grad_opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self._grad_opacity_slider.setRange(0, 100)
        self._grad_opacity_slider.setValue(100)
        self._lbl_grad_opacity = QLabel("100%")
        self._lbl_grad_opacity.setMinimumWidth(38)
        self._grad_opacity_slider.valueChanged.connect(
            lambda v: self._lbl_grad_opacity.setText(f"{v}%"))
        op_row.addWidget(self._grad_opacity_slider)
        op_row.addWidget(self._lbl_grad_opacity)
        grad_layout.addLayout(op_row)
        self._grad_box.setVisible(False)
        root.addWidget(self._grad_box)

        # ── Sezione Bacchetta magica ───────────────────────────────────
        self._wand_box = QGroupBox("Bacchetta magica")
        wand_layout = QVBoxLayout(self._wand_box)
        tol_row = QHBoxLayout()
        tol_row.addWidget(QLabel("Tolleranza:"))
        self._wand_tol_slider = QSlider(Qt.Orientation.Horizontal)
        self._wand_tol_slider.setRange(0, 255)
        self._wand_tol_slider.setValue(32)
        self._lbl_wand_tol = QLabel("32")
        self._lbl_wand_tol.setMinimumWidth(30)
        self._wand_tol_slider.valueChanged.connect(
            lambda v: self._lbl_wand_tol.setText(str(v)))
        tol_row.addWidget(self._wand_tol_slider)
        tol_row.addWidget(self._lbl_wand_tol)
        wand_layout.addLayout(tol_row)
        self._wand_contiguous = QCheckBox("Solo area contigua")
        self._wand_contiguous.setChecked(True)
        wand_layout.addWidget(self._wand_contiguous)
        self._wand_antialias = QCheckBox("Anti-alias bordi")
        self._wand_antialias.setChecked(True)
        wand_layout.addWidget(self._wand_antialias)
        self._wand_box.setVisible(False)
        root.addWidget(self._wand_box)

        # ── Sezione Forme (visibile solo con tool "shape") ─────────────
        self._shape_box = QGroupBox("Forme vettoriali")
        sh = QVBoxLayout(self._shape_box)

        sh.addWidget(QLabel("Tipo:"))
        self._shape_kind_combo = QComboBox()
        _SHAPE_KINDS = [
            ("Ellisse / Cerchio",  "ellipse"),
            ("Rettangolo",         "rect"),
            ("Poligono regolare",  "polygon"),
            ("Stella",             "star"),
            ("Squircle",           "squircle"),
            ("Freccia",            "arrow"),
            ("Logo SVG",           "svg"),
        ]
        for label, key in _SHAPE_KINDS:
            self._shape_kind_combo.addItem(label, key)
        self._shape_kind_combo.currentIndexChanged.connect(self._on_shape_kind_idx_changed)
        sh.addWidget(self._shape_kind_combo)

        # Colore riempimento
        fill_row = QHBoxLayout()
        fill_row.addWidget(QLabel("Riempimento:"))
        self._shape_fill_btn = QPushButton()
        self._shape_fill_btn.setFixedSize(36, 22)
        self._shape_fill_color = QColor(100, 149, 237, 200)
        self._shape_fill_btn.setStyleSheet(
            f"background:rgba(100,149,237,200); border:1px solid #666; border-radius:3px;")
        self._shape_fill_btn.clicked.connect(self._pick_shape_fill)
        self._shape_chk_fill = QCheckBox("Vis.")
        self._shape_chk_fill.setChecked(True)
        self._shape_chk_fill.toggled.connect(self._emit_shape_changed)
        fill_row.addWidget(self._shape_fill_btn)
        fill_row.addWidget(self._shape_chk_fill)
        fill_row.addStretch()
        sh.addLayout(fill_row)

        # Colore bordo + spessore
        stroke_row = QHBoxLayout()
        stroke_row.addWidget(QLabel("Bordo:"))
        self._shape_stroke_btn = QPushButton()
        self._shape_stroke_btn.setFixedSize(36, 22)
        self._shape_stroke_color = QColor(30, 30, 30, 255)
        self._shape_stroke_btn.setStyleSheet(
            "background:rgba(30,30,30,255); border:1px solid #666; border-radius:3px;")
        self._shape_stroke_btn.clicked.connect(self._pick_shape_stroke)
        self._shape_stroke_w = QSpinBox()
        self._shape_stroke_w.setRange(0, 60)
        self._shape_stroke_w.setValue(2)
        self._shape_stroke_w.setSuffix(" px")
        self._shape_stroke_w.setFixedWidth(68)
        self._shape_stroke_w.valueChanged.connect(self._emit_shape_changed)
        stroke_row.addWidget(self._shape_stroke_btn)
        stroke_row.addWidget(self._shape_stroke_w)
        stroke_row.addStretch()
        sh.addLayout(stroke_row)

        # Opacità
        op_sh_row = QHBoxLayout()
        op_sh_row.addWidget(QLabel("Opacità:"))
        self._shape_opacity_sl = QSlider(Qt.Orientation.Horizontal)
        self._shape_opacity_sl.setRange(0, 100)
        self._shape_opacity_sl.setValue(100)
        self._lbl_shape_op = QLabel("100%")
        self._lbl_shape_op.setFixedWidth(36)
        self._shape_opacity_sl.valueChanged.connect(
            lambda v: (self._lbl_shape_op.setText(f"{v}%"), self._emit_shape_changed()))
        op_sh_row.addWidget(self._shape_opacity_sl, 1)
        op_sh_row.addWidget(self._lbl_shape_op)
        sh.addLayout(op_sh_row)

        # Parametri specifici (visibili condizionalmente)
        # Angoli arrotondati (rect)
        self._sh_corner_row = QHBoxLayout()
        self._sh_corner_lbl = QLabel("Arrotondamento:")
        self._sh_corner_r = QSpinBox()
        self._sh_corner_r.setRange(0, 200); self._sh_corner_r.setValue(0)
        self._sh_corner_r.setSuffix(" px")
        self._sh_corner_r.valueChanged.connect(self._emit_shape_changed)
        self._sh_corner_row.addWidget(self._sh_corner_lbl)
        self._sh_corner_row.addWidget(self._sh_corner_r)
        self._sh_corner_row.addStretch()
        sh.addLayout(self._sh_corner_row)

        # Lati / punte (polygon, star)
        self._sh_sides_row = QHBoxLayout()
        self._sh_sides_lbl = QLabel("Lati / punte:")
        self._sh_n_sides = QSpinBox()
        self._sh_n_sides.setRange(3, 20); self._sh_n_sides.setValue(6)
        self._sh_n_sides.valueChanged.connect(self._emit_shape_changed)
        self._sh_sides_row.addWidget(self._sh_sides_lbl)
        self._sh_sides_row.addWidget(self._sh_n_sides)
        self._sh_sides_row.addStretch()
        sh.addLayout(self._sh_sides_row)

        # Raggio interno (star)
        self._sh_inner_row = QHBoxLayout()
        self._sh_inner_lbl = QLabel("Raggio interno:")
        self._sh_inner = QDoubleSpinBox()
        self._sh_inner.setRange(0.05, 0.95); self._sh_inner.setSingleStep(0.05)
        self._sh_inner.setValue(0.45)
        self._sh_inner.valueChanged.connect(self._emit_shape_changed)
        self._sh_inner_row.addWidget(self._sh_inner_lbl)
        self._sh_inner_row.addWidget(self._sh_inner)
        self._sh_inner_row.addStretch()
        sh.addLayout(self._sh_inner_row)

        # Esponente squircle
        self._sh_exp_row = QHBoxLayout()
        self._sh_exp_lbl = QLabel("Esponente:")
        self._sh_exp = QDoubleSpinBox()
        self._sh_exp.setRange(2.0, 20.0); self._sh_exp.setSingleStep(0.5)
        self._sh_exp.setValue(4.0)
        self._sh_exp.valueChanged.connect(self._emit_shape_changed)
        self._sh_exp_row.addWidget(self._sh_exp_lbl)
        self._sh_exp_row.addWidget(self._sh_exp)
        self._sh_exp_row.addStretch()
        sh.addLayout(self._sh_exp_row)

        # Freccia: larghezza gambo
        self._sh_aw_row = QHBoxLayout()
        self._sh_aw_lbl = QLabel("Largh. gambo:")
        self._sh_arrow_w = QSpinBox()
        self._sh_arrow_w.setRange(4, 200); self._sh_arrow_w.setValue(22)
        self._sh_arrow_w.setSuffix(" px")
        self._sh_arrow_w.valueChanged.connect(self._emit_shape_changed)
        self._sh_aw_row.addWidget(self._sh_aw_lbl)
        self._sh_aw_row.addWidget(self._sh_arrow_w)
        self._sh_aw_row.addStretch()
        sh.addLayout(self._sh_aw_row)

        # SVG: pulsante carica + tint
        self._sh_svg_row = QHBoxLayout()
        self._sh_btn_svg = QPushButton("📂 Carica SVG…")
        self._sh_btn_svg.clicked.connect(self._load_svg)
        self._sh_lbl_svg = QLabel("Nessun file")
        self._sh_lbl_svg.setStyleSheet("color:#888; font-size:10px;")
        self._sh_svg_row.addWidget(self._sh_btn_svg)
        sh.addLayout(self._sh_svg_row)
        sh.addWidget(self._sh_lbl_svg)

        self._sh_tint_row = QHBoxLayout()
        self._sh_chk_tint = QCheckBox("Tinta colore riempimento")
        self._sh_chk_tint.setChecked(False)
        self._sh_chk_tint.setToolTip(
            "Se attivo, sovrappone il colore di riempimento all'SVG.\n"
            "Utile per SVG monocromatici (loghi, icone).\n"
            "Disattiva per SVG multicolore.")
        self._sh_chk_tint.toggled.connect(self._emit_shape_changed)
        self._sh_tint_row.addWidget(self._sh_chk_tint)
        sh.addLayout(self._sh_tint_row)

        # Hint
        sh_hint = QLabel("<i>Clicca sul canvas per inserire.<br>"
                         "Trascina gli handle per modificare.</i>")
        sh_hint.setWordWrap(True)
        sh_hint.setStyleSheet("color:#888; font-size:10px;")
        sh.addWidget(sh_hint)

        # Applica
        sh_apply = QPushButton("✓ Applica forma sul layer")
        sh_apply.setStyleSheet("font-weight:bold; padding:5px;")
        sh_apply.clicked.connect(self.shape_apply)
        sh.addWidget(sh_apply)

        self._shape_box.setVisible(False)
        root.addWidget(self._shape_box)
        self._on_shape_kind_idx_changed(0)   # init visibilità parametri

        root.addStretch()

    # ── Public API ────────────────────────────────────────────────────

    @property
    def active_tool(self) -> str:
        return self._active_tool

    def current_color(self) -> QColor:
        return self._swatch.color()

    def set_color(self, c: QColor) -> None:
        self._swatch.set_color(c)
        self._update_rgba_label(c)

    @property
    def brush_size(self) -> int: return self._size_slider.value()
    @property
    def hardness(self) -> int:   return self._hard_slider.value()
    @property
    def opacity(self) -> int:    return self._op_slider.value()

    @property
    def gradient_color2(self) -> QColor:
        return self._swatch2.color() if hasattr(self, "_swatch2") else QColor(255, 255, 255, 255)
    @property
    def gradient_mode(self) -> str:
        if hasattr(self, "_combo_grad_mode"):
            return self._combo_grad_mode.currentData()
        return "linear"
    @property
    def gradient_opacity(self) -> float:
        if hasattr(self, "_grad_opacity_slider"):
            return self._grad_opacity_slider.value() / 100.0
        return 1.0

    @property
    def wand_tolerance(self) -> int:
        return self._wand_tol_slider.value() if hasattr(self, "_wand_tol_slider") else 32
    @property
    def wand_contiguous(self) -> bool:
        return self._wand_contiguous.isChecked() if hasattr(self, "_wand_contiguous") else True
    @property
    def wand_anti_alias(self) -> bool:
        return self._wand_antialias.isChecked() if hasattr(self, "_wand_antialias") else True

    @property
    def text_content(self) -> str: return self._txt_input.toPlainText()
    @property
    def font_family(self) -> str:  return self._font_combo.currentFont().family()
    @property
    def font_size_px(self) -> int: return self._font_size.value()
    @property
    def font_bold(self) -> bool:   return self._chk_bold.isChecked()
    @property
    def font_italic(self) -> bool: return self._chk_italic.isChecked()
    @property
    def text_opacity(self) -> float:
        return self._txt_opacity.value() / 100.0
    @property
    def text_effect(self) -> str:
        _map = {0: "none", 1: "shadow", 2: "outline", 3: "gradient",
                4: "arch", 5: "wave"}
        return _map.get(self._txt_effect.currentIndex(), "none")
    @property
    def text_effect_color(self) -> tuple:
        c = self._effect_color_val
        return (c.red(), c.green(), c.blue(), c.alpha())
    @property
    def text_outline_width(self) -> int:
        return self._txt_outline_w.value()

    # ── Brush / tip / rotation properties ────────────────────────────

    @property
    def brush_type(self) -> str:
        """Ritorna il tipo di pennello corrente."""
        _types = ["brush", "airbrush", "pencil", "calligraphy"]
        idx = self._brush_type_combo.currentIndex() if hasattr(self, "_brush_type_combo") else 0
        return _types[idx] if 0 <= idx < len(_types) else "brush"

    @property
    def tip_angle(self) -> int:
        return self._calli_angle.value() if hasattr(self, "_calli_angle") else 45

    @property
    def tip_ratio(self) -> int:
        return self._calli_ratio.value() if hasattr(self, "_calli_ratio") else 25

    @property
    def text_rotation(self) -> int:
        return self._txt_rotation.value() if hasattr(self, "_txt_rotation") else 0

    def set_text_rotation(self, deg: int) -> None:
        """Aggiorna lo spinner di rotazione testo senza emettere segnali."""
        if hasattr(self, "_txt_rotation"):
            self._txt_rotation.blockSignals(True)
            # Normalizza in [-180, 180] — corrisponde al range dello spinner
            v = ((float(deg) + 180.0) % 360.0) - 180.0
            self._txt_rotation.setValue(int(round(v)))
            self._txt_rotation.blockSignals(False)

    # ── Shape properties ──────────────────────────────────────────────
    def shape_kind_key(self) -> str:
        return self._shape_kind_combo.currentData() or "ellipse"

    @property
    def shape_fill_color(self) -> tuple:
        c = self._shape_fill_color
        a = c.alpha() if self._shape_chk_fill.isChecked() else 0
        return (c.red(), c.green(), c.blue(), a)

    @property
    def shape_stroke_color(self) -> tuple:
        c = self._shape_stroke_color
        return (c.red(), c.green(), c.blue(), c.alpha())

    @property
    def shape_stroke_width(self) -> int:
        return self._shape_stroke_w.value()

    @property
    def shape_opacity(self) -> float:
        return self._shape_opacity_sl.value() / 100.0

    @property
    def shape_n_sides(self) -> int:
        return self._sh_n_sides.value()

    @property
    def shape_inner_ratio(self) -> float:
        return self._sh_inner.value()

    @property
    def shape_sq_exp(self) -> float:
        return self._sh_exp.value()

    @property
    def shape_corner_r(self) -> float:
        return float(self._sh_corner_r.value())

    @property
    def shape_arrow_width(self) -> float:
        return float(self._sh_arrow_w.value())

    @property
    def shape_svg_tint(self) -> bool:
        return self._sh_chk_tint.isChecked()

    def get_svg_path(self) -> str:
        return getattr(self, '_svg_path_val', '')

    def set_svg_filename(self, name: str) -> None:
        self._sh_lbl_svg.setText(name)

    def sync_shape_from(self, shape) -> None:
        """Aggiorna i widget leggendo i valori dalla shape (senza emettere segnali)."""
        for widget in (self._sh_n_sides, self._sh_inner, self._sh_exp,
                       self._sh_corner_r, self._sh_arrow_w, self._shape_opacity_sl):
            widget.blockSignals(True)
        self._sh_n_sides.setValue(shape.n_sides)
        self._sh_inner.setValue(shape.inner_ratio)
        self._sh_exp.setValue(shape.sq_exp)
        self._sh_corner_r.setValue(int(shape.corner_r))
        self._sh_arrow_w.setValue(int(shape.arrow_width))
        self._shape_opacity_sl.setValue(int(shape.style.opacity * 100))
        for widget in (self._sh_n_sides, self._sh_inner, self._sh_exp,
                       self._sh_corner_r, self._sh_arrow_w, self._shape_opacity_sl):
            widget.blockSignals(False)

    def select_tool(self, key: str) -> None:
        if key in self._btns:
            same = (self._active_tool == key)
            self._activate(None if same else key)

    # ── Slots ─────────────────────────────────────────────────────────

    def _on_btn_clicked(self, key: str, checked: bool) -> None:
        if not checked or self._active_tool == key:
            self._activate(None)
        else:
            self._activate(key)

    def _on_color_changed(self, c: QColor) -> None:
        self._update_rgba_label(c)
        self.color_changed.emit(c)

    def _on_text_params_changed(self, *_) -> None:
        self.text_params_changed.emit()

    def _on_text_effect_changed(self, idx: int) -> None:
        # Mostra/nascondi i controlli secondari in base all'effetto scelto
        has_color   = idx in (1, 2, 3)   # ombra, bordo, gradiente
        has_outline = idx == 2            # solo bordo
        self._lbl_effect_color.setVisible(has_color)
        self._btn_effect_color.setVisible(has_color)
        self._txt_outline_w.setVisible(has_outline)
        # il label "Spessore bordo" è il primo widget del _outline_row
        for i in range(self._outline_row.count()):
            w = self._outline_row.itemAt(i).widget()
            if w:
                w.setVisible(has_outline)
        self._on_text_params_changed()

    def _pick_effect_color(self) -> None:
        c = QColorDialog.getColor(
            self._effect_color_val, self, "Colore effetto",
            QColorDialog.ColorDialogOption.ShowAlphaChannel)
        if c.isValid():
            self._effect_color_val = c
            self._btn_effect_color.setStyleSheet(
                f"background:{c.name()}; border:1px solid #666;")
            self._on_text_params_changed()

    # ── Helpers ───────────────────────────────────────────────────────

    def _activate(self, key: str | None) -> None:
        self._active_tool = key or ""
        for k, btn in self._btns.items():
            btn.setChecked(k == self._active_tool)

        # Mostra/nascondi sezioni contestuali
        is_draw   = self._active_tool in ("brush", "eraser", "smudge", "heal")
        is_clone  = self._active_tool == "clone"
        is_text   = self._active_tool == "text"
        is_eye    = self._active_tool == "eyedropper"
        is_select = self._active_tool in ("select_rect", "select_ellipse", "select_lasso")
        is_grad   = self._active_tool == "gradient"
        is_wand   = self._active_tool == "magic_wand"
        is_shape  = self._active_tool == "shape"

        self._size_box.setVisible(is_draw or is_clone)
        self._hard_box.setVisible(is_draw or is_clone)
        self._op_box.setVisible((is_draw or is_clone) and self._active_tool != "smudge")
        self._text_box.setVisible(is_text)
        if hasattr(self, "_brush_type_box"):
            # Mostra solo per brush (non eraser/smudge che hanno logica propria)
            self._brush_type_box.setVisible(self._active_tool == "brush")
        if hasattr(self, "_sel_box"):
            self._sel_box.setVisible(is_select)
        if hasattr(self, "_clone_hint"):
            self._clone_hint.setVisible(is_clone)
        if hasattr(self, "_grad_box"):
            self._grad_box.setVisible(is_grad)
        if hasattr(self, "_wand_box"):
            self._wand_box.setVisible(is_wand)
        if hasattr(self, "_shape_box"):
            self._shape_box.setVisible(is_shape)

        if self._active_tool == "smudge":
            self._hard_box.setTitle("Intensità sfumino")
        elif self._active_tool == "heal":
            self._hard_box.setTitle("Morbidezza guarigione")
        elif is_clone:
            self._hard_box.setTitle("Morbidezza")
        else:
            self._hard_box.setTitle("Durezza")

        self.tool_changed.emit(self._active_tool)

    def _emit_shape_changed(self, *_) -> None:
        self.shape_changed.emit()

    def _on_brush_type_changed(self, idx: int) -> None:
        is_calli = (idx == 3)   # "Calligrafia"
        for row in (self._calli_angle_row, self._calli_ratio_row):
            for i in range(row.count()):
                w = row.itemAt(i).widget()
                if w:
                    w.setVisible(is_calli)

    def _on_shape_kind_idx_changed(self, idx: int) -> None:
        key = self._shape_kind_combo.itemData(idx) or "ellipse"
        is_rect  = key == "rect"
        is_poly  = key == "polygon"
        is_star  = key == "star"
        is_sq    = key == "squircle"
        is_arrow = key == "arrow"
        is_svg   = key == "svg"
        is_ps    = is_poly or is_star

        def _sv(layout, visible):
            for i in range(layout.count()):
                w = layout.itemAt(i).widget()
                if w:
                    w.setVisible(visible)

        _sv(self._sh_corner_row, is_rect)
        _sv(self._sh_sides_row,  is_ps)
        _sv(self._sh_inner_row,  is_star)
        _sv(self._sh_exp_row,    is_sq)
        _sv(self._sh_aw_row,     is_arrow)
        self._sh_btn_svg.setVisible(is_svg)
        self._sh_lbl_svg.setVisible(is_svg)
        _sv(self._sh_tint_row, is_svg)

        self.shape_kind_changed.emit(key)
        self._emit_shape_changed()

    def _pick_shape_fill(self) -> None:
        c = QColorDialog.getColor(self._shape_fill_color, self, "Colore riempimento",
                                   QColorDialog.ColorDialogOption.ShowAlphaChannel)
        if c.isValid():
            self._shape_fill_color = c
            self._shape_fill_btn.setStyleSheet(
                f"background:rgba({c.red()},{c.green()},{c.blue()},{c.alpha()});"
                "border:1px solid #666; border-radius:3px;")
            self._emit_shape_changed()

    def _pick_shape_stroke(self) -> None:
        c = QColorDialog.getColor(self._shape_stroke_color, self, "Colore bordo",
                                   QColorDialog.ColorDialogOption.ShowAlphaChannel)
        if c.isValid():
            self._shape_stroke_color = c
            self._shape_stroke_btn.setStyleSheet(
                f"background:rgba({c.red()},{c.green()},{c.blue()},{c.alpha()});"
                "border:1px solid #666; border-radius:3px;")
            self._emit_shape_changed()

    def _load_svg(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Carica logo SVG", "",
            "File SVG (*.svg *.svgz);;Tutti i file (*)")
        if path:
            self._svg_path_val = path
            from pathlib import Path
            self._sh_lbl_svg.setText(Path(path).name)
            self._emit_shape_changed()

    def _update_rgba_label(self, c: QColor) -> None:
        self._lbl_rgba.setText(f"R:{c.red()} G:{c.green()} B:{c.blue()} A:{c.alpha()}")

    @staticmethod
    def _make_slider_row(parent: QGroupBox, lo: int, hi: int, val: int, fmt):
        layout = QVBoxLayout(parent)
        row = QHBoxLayout()
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(lo, hi); slider.setValue(val)
        lbl = QLabel(fmt(val)); lbl.setMinimumWidth(52)
        slider.valueChanged.connect(lambda v, l=lbl, f=fmt: l.setText(f(v)))
        row.addWidget(slider); row.addWidget(lbl)
        layout.addLayout(row)
        return slider, lbl
