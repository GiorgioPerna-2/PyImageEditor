from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider,
    QPushButton, QGroupBox, QComboBox, QScrollArea,
)

from core.image_filters import COLOR_STYLES


def _make_slider(min_: int, max_: int, default: int = 0) -> QSlider:
    s = QSlider(Qt.Orientation.Horizontal)
    s.setMinimum(min_)
    s.setMaximum(max_)
    s.setValue(default)
    return s


def _labeled_slider(layout: QVBoxLayout, label: str,
                    min_: int, max_: int, default: int = 0) -> QSlider:
    row = QHBoxLayout()
    lbl = QLabel(label)
    lbl.setMinimumWidth(80)
    val_lbl = QLabel(str(default))
    val_lbl.setMinimumWidth(32)
    val_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
    s = _make_slider(min_, max_, default)
    s.valueChanged.connect(lambda v: val_lbl.setText(str(v)))
    row.addWidget(lbl)
    row.addWidget(s)
    row.addWidget(val_lbl)
    layout.addLayout(row)
    return s


class FilterPanel(QWidget):
    filters_changed = pyqtSignal(dict)
    apply_clicked = pyqtSignal()
    reset_clicked = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()

        # Stato corrente — chiavi devono corrispondere a FilterSettings
        self.values: dict = {
            # Tonali
            "brightness": 0, "contrast": 0, "saturation": 0,
            "sharpen": 0, "blur": 0, "hue_shift": 0, "vibrance": 0,
            # Stile
            "color_style": "none", "color_style_intensity": 100,
            # Effetti
            "vignette": 0, "grain": 0, "glow": 0,
            "clarity": 0, "dehaze": 0,
        }

        # Layout esterno: scroll area per pannello alto
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        outer.addWidget(scroll)

        content = QWidget()
        main = QVBoxLayout(content)
        main.setContentsMargins(8, 8, 8, 8)
        main.setSpacing(10)
        scroll.setWidget(content)

        self._sliders: dict[str, QSlider] = {}

        # ── Sezione Tonali ────────────────────────────────────────────
        tonal_group = QGroupBox("Tonali")
        tonal_layout = QVBoxLayout(tonal_group)
        for name, mn, mx in [
            ("brightness",  -100, 100),
            ("contrast",    -100, 100),
            ("saturation",  -100, 100),
            ("vibrance",    -100, 100),
            ("hue_shift",   -180, 180),
            ("sharpen",        0, 100),
            ("blur",           0,  20),
        ]:
            label = {
                "brightness": "Luminosità", "contrast": "Contrasto",
                "saturation": "Saturazione", "vibrance": "Vibrance",
                "hue_shift": "Tinta", "sharpen": "Nitidezza", "blur": "Sfocatura",
            }[name]
            s = _labeled_slider(tonal_layout, label, mn, mx, 0)
            s.valueChanged.connect(lambda v, n=name: self._on_slider(n, v))
            self._sliders[name] = s
        main.addWidget(tonal_group)

        # ── Sezione Stile Colore ──────────────────────────────────────
        style_group = QGroupBox("Stile Colore")
        style_layout = QVBoxLayout(style_group)

        self._combo_style = QComboBox()
        for key, (label, _) in COLOR_STYLES.items():
            self._combo_style.addItem(label, userData=key)
        self._combo_style.currentIndexChanged.connect(self._on_style_changed)
        style_layout.addWidget(self._combo_style)

        self._lbl_intensity = QLabel("Intensità")
        style_layout.addWidget(self._lbl_intensity)
        s_intensity = _labeled_slider(style_layout, "Intensità", 0, 100, 100)
        s_intensity.valueChanged.connect(lambda v: self._on_slider("color_style_intensity", v))
        self._sliders["color_style_intensity"] = s_intensity
        main.addWidget(style_group)

        # ── Sezione Effetti Fotografici ───────────────────────────────
        fx_group = QGroupBox("Effetti Fotografici")
        fx_layout = QVBoxLayout(fx_group)
        for name, mn, mx in [
            ("vignette",  0, 100),
            ("grain",     0, 100),
            ("glow",      0, 100),
            ("clarity", -100, 100),
            ("dehaze",    0, 100),
        ]:
            label = {
                "vignette": "Vignettatura", "grain": "Grana",
                "glow": "Bagliore", "clarity": "Clarity",
                "dehaze": "Dehaze",
            }[name]
            s = _labeled_slider(fx_layout, label, mn, mx, 0)
            s.valueChanged.connect(lambda v, n=name: self._on_slider(n, v))
            self._sliders[name] = s
        main.addWidget(fx_group)

        # ── Pulsanti Apply / Reset ────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_apply = QPushButton("✓ Applica")
        btn_apply.setStyleSheet("font-weight: bold;")
        btn_reset = QPushButton("↺ Reset")
        btn_apply.clicked.connect(self.apply_clicked.emit)
        btn_reset.clicked.connect(self._reset)
        btn_row.addWidget(btn_apply)
        btn_row.addWidget(btn_reset)
        main.addLayout(btn_row)
        main.addStretch()

    # ------------------------------------------------------------------

    def _on_slider(self, name: str, value: int) -> None:
        self.values[name] = value
        self.filters_changed.emit(self.values.copy())

    def _on_style_changed(self, _: int) -> None:
        key = self._combo_style.currentData()
        self.values["color_style"] = key
        self.filters_changed.emit(self.values.copy())

    def _reset(self) -> None:
        """Azzera tutti i controlli senza emettere segnali intermedi."""
        for name, s in self._sliders.items():
            s.blockSignals(True)
            s.setValue(0 if name != "color_style_intensity" else 100)
            s.blockSignals(False)
        self._combo_style.blockSignals(True)
        self._combo_style.setCurrentIndex(0)
        self._combo_style.blockSignals(False)
        self.values = {k: (100 if k == "color_style_intensity" else (0 if k != "color_style" else "none"))
                       for k in self.values}
        self.reset_clicked.emit()
