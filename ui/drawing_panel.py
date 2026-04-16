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
    QFontComboBox, QSpinBox, QCheckBox, QScrollArea, QComboBox,
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

        # Contenuto
        self._txt_input = QLineEdit()
        self._txt_input.setPlaceholderText("Scrivi qui il testo…")
        self._txt_input.textChanged.connect(self._on_text_params_changed)
        text_layout.addWidget(QLabel("Contenuto:"))
        text_layout.addWidget(self._txt_input)

        # Font family
        self._font_combo = QFontComboBox()
        self._font_combo.setCurrentFont(QFont("Arial"))
        self._font_combo.currentFontChanged.connect(self._on_text_params_changed)
        text_layout.addWidget(QLabel("Font:"))
        text_layout.addWidget(self._font_combo)

        # Size + Bold + Italic
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

        # Istruzioni
        hint = QLabel("<i>Clicca sul canvas per posizionare il testo.<br>"
                      "Trascina per spostarlo.</i>")
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
    def text_content(self) -> str: return self._txt_input.text()
    @property
    def font_family(self) -> str:  return self._font_combo.currentFont().family()
    @property
    def font_size_px(self) -> int: return self._font_size.value()
    @property
    def font_bold(self) -> bool:   return self._chk_bold.isChecked()
    @property
    def font_italic(self) -> bool: return self._chk_italic.isChecked()

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

        self._size_box.setVisible(is_draw or is_clone)
        self._hard_box.setVisible(is_draw or is_clone)
        self._op_box.setVisible((is_draw or is_clone) and self._active_tool != "smudge")
        self._text_box.setVisible(is_text)
        if hasattr(self, "_sel_box"):
            self._sel_box.setVisible(is_select)
        if hasattr(self, "_clone_hint"):
            self._clone_hint.setVisible(is_clone)
        if hasattr(self, "_grad_box"):
            self._grad_box.setVisible(is_grad)
        if hasattr(self, "_wand_box"):
            self._wand_box.setVisible(is_wand)

        if self._active_tool == "smudge":
            self._hard_box.setTitle("Intensità sfumino")
        elif self._active_tool == "heal":
            self._hard_box.setTitle("Morbidezza guarigione")
        elif is_clone:
            self._hard_box.setTitle("Morbidezza")
        else:
            self._hard_box.setTitle("Durezza")

        self.tool_changed.emit(self._active_tool)

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
