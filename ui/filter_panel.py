from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider,
    QPushButton, QGroupBox, QComboBox, QScrollArea,
    QDoubleSpinBox, QSpinBox, QCheckBox,
)

from core.image_filters import COLOR_STYLES, SPATIAL_FILTERS


def _make_slider(min_: int, max_: int, default: int = 0) -> QSlider:
    s = QSlider(Qt.Orientation.Horizontal)
    s.setMinimum(min_); s.setMaximum(max_); s.setValue(default)
    return s


def _labeled_slider(layout, label: str, min_: int, max_: int,
                     default: int = 0) -> QSlider:
    row = QHBoxLayout()
    lbl = QLabel(label); lbl.setMinimumWidth(80)
    val_lbl = QLabel(str(default))
    val_lbl.setMinimumWidth(32)
    val_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
    s = _make_slider(min_, max_, default)
    s.valueChanged.connect(lambda v: val_lbl.setText(str(v)))
    row.addWidget(lbl); row.addWidget(s); row.addWidget(val_lbl)
    layout.addLayout(row)
    return s


class FilterPanel(QWidget):
    filters_changed  = pyqtSignal(dict)
    apply_clicked    = pyqtSignal()
    reset_clicked    = pyqtSignal()
    # Emesso quando si richiede un filtro spaziale/FFT
    # args: (filter_key, params_dict, apply_to_selection_only)
    spatial_apply    = pyqtSignal(str, dict, bool)
    # Richiesta visualizzazione spettro FFT
    spectrum_request = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()

        self.values: dict = {
            "brightness": 0, "contrast": 0, "saturation": 0,
            "sharpen": 0, "blur": 0, "hue_shift": 0, "vibrance": 0,
            "color_style": "none", "color_style_intensity": 100,
            "vignette": 0, "grain": 0, "glow": 0, "clarity": 0, "dehaze": 0,
        }

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

        # ── Tonali ──────────────────────────────────────────────────
        tonal_group = QGroupBox("Tonali")
        tl = QVBoxLayout(tonal_group)
        for name, mn, mx in [
            ("brightness", -100, 100), ("contrast",   -100, 100),
            ("saturation", -100, 100), ("vibrance",   -100, 100),
            ("hue_shift",  -180, 180), ("sharpen",       0, 100),
            ("blur",          0,  20),
        ]:
            label = {
                "brightness": "Luminosità", "contrast": "Contrasto",
                "saturation": "Saturazione", "vibrance": "Vibrance",
                "hue_shift": "Tinta", "sharpen": "Nitidezza", "blur": "Sfocatura",
            }[name]
            s = _labeled_slider(tl, label, mn, mx, 0)
            s.valueChanged.connect(lambda v, n=name: self._on_slider(n, v))
            self._sliders[name] = s
        main.addWidget(tonal_group)

        # ── Stile Colore ─────────────────────────────────────────────
        style_group = QGroupBox("Stile Colore")
        sl = QVBoxLayout(style_group)
        self._combo_style = QComboBox()
        for key, (label, _) in COLOR_STYLES.items():
            self._combo_style.addItem(label, userData=key)
        self._combo_style.currentIndexChanged.connect(self._on_style_changed)
        sl.addWidget(self._combo_style)
        s_int = _labeled_slider(sl, "Intensità", 0, 100, 100)
        s_int.valueChanged.connect(lambda v: self._on_slider("color_style_intensity", v))
        self._sliders["color_style_intensity"] = s_int
        main.addWidget(style_group)

        # ── Effetti Fotografici ───────────────────────────────────────
        fx_group = QGroupBox("Effetti Fotografici")
        fl = QVBoxLayout(fx_group)
        for name, mn, mx in [
            ("vignette", 0, 100), ("grain", 0, 100),
            ("glow", 0, 100), ("clarity", -100, 100), ("dehaze", 0, 100),
        ]:
            label = {
                "vignette": "Vignettatura", "grain": "Grana",
                "glow": "Bagliore", "clarity": "Clarity", "dehaze": "Dehaze",
            }[name]
            s = _labeled_slider(fl, label, mn, mx, 0)
            s.valueChanged.connect(lambda v, n=name: self._on_slider(n, v))
            self._sliders[name] = s
        main.addWidget(fx_group)

        # ── Filtri Spaziali & FFT ─────────────────────────────────────
        sp_group = QGroupBox("Filtri Spaziali / FFT")
        sp_layout = QVBoxLayout(sp_group)

        self._spatial_combo = QComboBox()
        for key, (label, _, _) in SPATIAL_FILTERS.items():
            self._spatial_combo.addItem(label, userData=key)
        self._spatial_combo.currentIndexChanged.connect(self._on_spatial_changed)
        sp_layout.addWidget(self._spatial_combo)

        # Parametri dinamici (mostrati/nascosti in base al filtro)
        self._sp_params_widget = QWidget()
        sp_params_layout = QVBoxLayout(self._sp_params_widget)
        sp_params_layout.setContentsMargins(0, 0, 0, 0)

        # radius (int)
        self._sp_radius_row = QHBoxLayout()
        self._sp_radius_row.addWidget(QLabel("Raggio:"))
        self._sp_radius = QSpinBox(); self._sp_radius.setRange(1, 20); self._sp_radius.setValue(3)
        self._sp_radius.setSuffix(" px")
        self._sp_radius_row.addWidget(self._sp_radius); self._sp_radius_row.addStretch()
        sp_params_layout.addLayout(self._sp_radius_row)

        # amount (float)
        self._sp_amount_row = QHBoxLayout()
        self._sp_amount_row.addWidget(QLabel("Quantità:"))
        self._sp_amount = QDoubleSpinBox()
        self._sp_amount.setRange(0.0, 5.0); self._sp_amount.setSingleStep(0.1)
        self._sp_amount.setValue(1.0); self._sp_amount.setDecimals(2)
        self._sp_amount_row.addWidget(self._sp_amount); self._sp_amount_row.addStretch()
        sp_params_layout.addLayout(self._sp_amount_row)

        # sigma (int, per bilateral)
        self._sp_sigma_row = QHBoxLayout()
        self._sp_sigma_row.addWidget(QLabel("Sigma:"))
        self._sp_sigma = QSpinBox(); self._sp_sigma.setRange(1, 200); self._sp_sigma.setValue(50)
        self._sp_sigma_row.addWidget(self._sp_sigma); self._sp_sigma_row.addStretch()
        sp_params_layout.addLayout(self._sp_sigma_row)

        # direction (int, per emboss)
        self._sp_dir_row = QHBoxLayout()
        self._sp_dir_row.addWidget(QLabel("Direzione:"))
        self._sp_dir = QSpinBox(); self._sp_dir.setRange(0, 359); self._sp_dir.setValue(135)
        self._sp_dir.setSuffix("°")
        self._sp_dir_row.addWidget(self._sp_dir); self._sp_dir_row.addStretch()
        sp_params_layout.addLayout(self._sp_dir_row)

        # cutoff (float, per FFT low/high)
        self._sp_cutoff_row = QHBoxLayout()
        self._sp_cutoff_row.addWidget(QLabel("Taglio:"))
        self._sp_cutoff = QDoubleSpinBox()
        self._sp_cutoff.setRange(0.01, 0.99); self._sp_cutoff.setSingleStep(0.05)
        self._sp_cutoff.setValue(0.3); self._sp_cutoff.setDecimals(2)
        self._sp_cutoff_row.addWidget(self._sp_cutoff); self._sp_cutoff_row.addStretch()
        sp_params_layout.addLayout(self._sp_cutoff_row)

        # band low/high (float, per FFT band)
        self._sp_band_row = QHBoxLayout()
        self._sp_band_row.addWidget(QLabel("Banda:"))
        self._sp_band_low = QDoubleSpinBox()
        self._sp_band_low.setRange(0.0, 0.98); self._sp_band_low.setSingleStep(0.05)
        self._sp_band_low.setValue(0.1); self._sp_band_low.setDecimals(2)
        self._sp_band_row.addWidget(self._sp_band_low)
        self._sp_band_row.addWidget(QLabel("–"))
        self._sp_band_high = QDoubleSpinBox()
        self._sp_band_high.setRange(0.02, 1.0); self._sp_band_high.setSingleStep(0.05)
        self._sp_band_high.setValue(0.5); self._sp_band_high.setDecimals(2)
        self._sp_band_row.addWidget(self._sp_band_high); self._sp_band_row.addStretch()
        sp_params_layout.addLayout(self._sp_band_row)

        # notch radius (float, per FFT moire)
        self._sp_notch_row = QHBoxLayout()
        self._sp_notch_row.addWidget(QLabel("Notch R:"))
        self._sp_notch = QDoubleSpinBox()
        self._sp_notch.setRange(0.01, 0.2); self._sp_notch.setSingleStep(0.01)
        self._sp_notch.setValue(0.05); self._sp_notch.setDecimals(3)
        self._sp_notch_row.addWidget(self._sp_notch); self._sp_notch_row.addStretch()
        sp_params_layout.addLayout(self._sp_notch_row)

        sp_layout.addWidget(self._sp_params_widget)

        # Solo selezione
        sel_row = QHBoxLayout()
        self._sp_chk_sel = QCheckBox("Solo area selezionata")
        sel_row.addWidget(self._sp_chk_sel)
        sp_layout.addLayout(sel_row)

        # Pulsanti
        sp_btn_row = QHBoxLayout()
        self._sp_btn_apply = QPushButton("✓ Applica filtro")
        self._sp_btn_apply.setStyleSheet("font-weight:bold;")
        self._sp_btn_apply.clicked.connect(self._on_spatial_apply)
        sp_btn_row.addWidget(self._sp_btn_apply)

        self._sp_btn_spectrum = QPushButton("📊 Mostra spettro")
        self._sp_btn_spectrum.setToolTip("Visualizza lo spettro FFT dell'immagine")
        self._sp_btn_spectrum.clicked.connect(self.spectrum_request.emit)
        sp_btn_row.addWidget(self._sp_btn_spectrum)
        sp_layout.addLayout(sp_btn_row)
        main.addWidget(sp_group)

        # ── Pulsanti Applica / Reset (tonali) ────────────────────────
        btn_row = QHBoxLayout()
        btn_apply = QPushButton("✓ Applica tonali")
        btn_apply.setStyleSheet("font-weight:bold;")
        btn_reset = QPushButton("↺ Reset")
        btn_apply.clicked.connect(self.apply_clicked.emit)
        btn_reset.clicked.connect(self._reset)
        btn_row.addWidget(btn_apply); btn_row.addWidget(btn_reset)
        main.addLayout(btn_row)
        main.addStretch()

        # Init visibilità parametri
        self._on_spatial_changed(0)

    # ── Segnali ──────────────────────────────────────────────────────

    def _on_slider(self, name: str, value: int) -> None:
        self.values[name] = value
        self.filters_changed.emit(self.values.copy())

    def _on_style_changed(self, _: int) -> None:
        self.values["color_style"] = self._combo_style.currentData()
        self.filters_changed.emit(self.values.copy())

    def _on_spatial_changed(self, _: int = 0) -> None:
        key = self._spatial_combo.currentData() or "none"
        _, _, defaults = SPATIAL_FILTERS.get(key, ("", None, {}))

        has_radius  = "radius"    in defaults
        has_amount  = "amount"    in defaults
        has_sigma   = "sigma"     in defaults
        has_dir     = "direction" in defaults
        has_cutoff  = "cutoff"    in defaults
        has_band    = "low"       in defaults
        has_notch   = "radius"    in defaults and key == "fft_moire"
        is_spectrum = key == "fft_spectrum"
        is_none     = key == "none"

        def _sv(layout, visible):
            for i in range(layout.count()):
                item = layout.itemAt(i)
                w = item.widget() if item else None
                if w: w.setVisible(visible)

        _sv(self._sp_radius_row, has_radius and not has_notch)
        _sv(self._sp_amount_row, has_amount)
        _sv(self._sp_sigma_row,  has_sigma)
        _sv(self._sp_dir_row,    has_dir)
        _sv(self._sp_cutoff_row, has_cutoff)
        _sv(self._sp_band_row,   has_band)
        _sv(self._sp_notch_row,  has_notch)

        self._sp_btn_apply.setVisible(not is_spectrum and not is_none)
        self._sp_btn_spectrum.setVisible(True)
        self._sp_params_widget.setVisible(not is_spectrum and not is_none)

        if not is_none and not is_spectrum:
            if "radius"    in defaults: self._sp_radius.setValue(int(defaults["radius"]))
            if "amount"    in defaults: self._sp_amount.setValue(float(defaults["amount"]))
            if "sigma"     in defaults: self._sp_sigma.setValue(int(defaults["sigma"]))
            if "direction" in defaults: self._sp_dir.setValue(int(defaults["direction"]))
            if "cutoff"    in defaults: self._sp_cutoff.setValue(float(defaults["cutoff"]))
            if "low"       in defaults: self._sp_band_low.setValue(float(defaults["low"]))
            if "high"      in defaults: self._sp_band_high.setValue(float(defaults["high"]))

    def _on_spatial_apply(self) -> None:
        key = self._spatial_combo.currentData() or "none"
        if key == "none":
            return
        _, _, defaults = SPATIAL_FILTERS.get(key, ("", None, {}))
        params: dict = {}
        if "radius"    in defaults: params["radius"]    = self._sp_radius.value()
        if "amount"    in defaults: params["amount"]    = self._sp_amount.value()
        if "sigma"     in defaults: params["sigma"]     = self._sp_sigma.value()
        if "direction" in defaults: params["direction"] = self._sp_dir.value()
        if "cutoff"    in defaults: params["cutoff"]    = self._sp_cutoff.value()
        if "low"       in defaults: params["low"]       = self._sp_band_low.value()
        if "high"      in defaults: params["high"]      = self._sp_band_high.value()
        sel_only = self._sp_chk_sel.isChecked()
        self.spatial_apply.emit(key, params, sel_only)

    def _reset(self) -> None:
        for name, s in self._sliders.items():
            s.blockSignals(True)
            s.setValue(100 if name == "color_style_intensity" else 0)
            s.blockSignals(False)
        self._combo_style.blockSignals(True)
        self._combo_style.setCurrentIndex(0)
        self._combo_style.blockSignals(False)
        self.values = {k: (100 if k == "color_style_intensity" else
                           (0 if k != "color_style" else "none"))
                       for k in self.values}
        self.reset_clicked.emit()
