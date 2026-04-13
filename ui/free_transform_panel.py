"""
ui/free_transform_panel.py
--------------------------
Pannello trasformazione libera: scala, rotazione, skew con anteprima live.
La trasformazione viene applicata solo quando l'utente clicca "Applica".
"""
from __future__ import annotations
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider,
    QPushButton, QGroupBox, QDoubleSpinBox, QCheckBox,
)


class FreeTransformPanel(QWidget):

    preview_requested = pyqtSignal(dict)   # parametri per anteprima live
    apply_requested   = pyqtSignal(dict)   # applica definitivamente
    reset_requested   = pyqtSignal()       # torna all'originale

    def __init__(self) -> None:
        super().__init__()
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        info = QLabel(
            "Modifica i valori per trasformare il layer attivo.\n"
            "L'anteprima è visibile in tempo reale."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #aaa; font-size: 10px;")
        root.addWidget(info)

        # ── Scala ─────────────────────────────────────────────────────
        scale_grp = QGroupBox("Scala")
        sg = QVBoxLayout(scale_grp)
        self._chk_aspect = QCheckBox("Mantieni proporzioni")
        self._chk_aspect.setChecked(True)
        sg.addWidget(self._chk_aspect)
        self._spin_sx = self._spinbox(0.05, 10.0, 1.0, "×", step=0.05)
        self._spin_sy = self._spinbox(0.05, 10.0, 1.0, "×", step=0.05)
        self._spin_sx.valueChanged.connect(self._on_sx_changed)
        self._spin_sy.valueChanged.connect(self._on_sy_changed)
        self._add_row(sg, "Larghezza:", self._spin_sx)
        self._add_row(sg, "Altezza:", self._spin_sy)

        self._spin_w_pct = self._spinbox(1, 1000, 100, " %", decimals=0, step=1)
        self._spin_h_pct = self._spinbox(1, 1000, 100, " %", decimals=0, step=1)
        self._spin_w_pct.valueChanged.connect(self._on_wpct_changed)
        self._spin_h_pct.valueChanged.connect(self._on_hpct_changed)
        self._add_row(sg, "W %:", self._spin_w_pct)
        self._add_row(sg, "H %:", self._spin_h_pct)
        root.addWidget(scale_grp)

        # ── Rotazione ─────────────────────────────────────────────────
        rot_grp = QGroupBox("Rotazione")
        rg = QVBoxLayout(rot_grp)
        self._spin_angle = self._spinbox(-360.0, 360.0, 0.0, "°", step=0.5)
        self._spin_angle.valueChanged.connect(self._emit_preview)
        self._add_row(rg, "Angolo:", self._spin_angle)

        btn_row = QHBoxLayout()
        for label, deg in [("↺ 90°", -90), ("↻ 90°", 90), ("180°", 180)]:
            btn = QPushButton(label)
            btn.clicked.connect(lambda _, d=deg: self._set_angle(d))
            btn_row.addWidget(btn)
        rg.addLayout(btn_row)
        root.addWidget(rot_grp)

        # ── Skew ──────────────────────────────────────────────────────
        skew_grp = QGroupBox("Inclinazione (Skew)")
        sk = QVBoxLayout(skew_grp)
        self._spin_skx = self._spinbox(-45.0, 45.0, 0.0, "°", step=0.5)
        self._spin_sky = self._spinbox(-45.0, 45.0, 0.0, "°", step=0.5)
        self._spin_skx.valueChanged.connect(self._emit_preview)
        self._spin_sky.valueChanged.connect(self._emit_preview)
        self._add_row(sk, "Orizzontale:", self._spin_skx)
        self._add_row(sk, "Verticale:", self._spin_sky)
        root.addWidget(skew_grp)

        # ── Opzioni ───────────────────────────────────────────────────
        self._chk_expand = QCheckBox("Espandi canvas (non taglia)")
        self._chk_expand.setChecked(True)
        root.addWidget(self._chk_expand)

        # ── Pulsanti ──────────────────────────────────────────────────
        btn_row2 = QHBoxLayout()
        btn_apply = QPushButton("✓ Applica")
        btn_apply.setStyleSheet("font-weight: bold; background: #2d7d46; color: white; padding: 5px;")
        btn_apply.clicked.connect(self._on_apply)
        btn_reset = QPushButton("↺ Reset")
        btn_reset.clicked.connect(self._on_reset)
        btn_row2.addWidget(btn_apply)
        btn_row2.addWidget(btn_reset)
        root.addLayout(btn_row2)
        root.addStretch()

        self._blocking = False   # evita loop durante la sincronizzazione proporzioni

    # ── Helpers ───────────────────────────────────────────────────────

    def _spinbox(self, lo, hi, val, suffix="", decimals=2, step=0.1):
        s = QDoubleSpinBox()
        s.setRange(lo, hi); s.setValue(val)
        s.setDecimals(decimals if isinstance(lo, float) else 0)
        s.setSingleStep(step); s.setSuffix(suffix)
        return s

    def _add_row(self, layout, label, widget):
        r = QHBoxLayout()
        lbl = QLabel(label); lbl.setMinimumWidth(90)
        r.addWidget(lbl); r.addWidget(widget)
        layout.addLayout(r)

    def _set_angle(self, delta: float) -> None:
        self._spin_angle.setValue(self._spin_angle.value() + delta)

    def _current_params(self) -> dict:
        return {
            "scale_x": self._spin_sx.value(),
            "scale_y": self._spin_sy.value(),
            "angle_deg": self._spin_angle.value(),
            "skew_x": self._spin_skx.value(),
            "skew_y": self._spin_sky.value(),
            "expand": self._chk_expand.isChecked(),
        }

    # ── Sincronizzazione proporzioni ──────────────────────────────────

    def _on_sx_changed(self, v: float) -> None:
        if self._blocking: return
        self._blocking = True
        self._spin_w_pct.setValue(v * 100)
        if self._chk_aspect.isChecked():
            self._spin_sy.setValue(v)
            self._spin_h_pct.setValue(v * 100)
        self._blocking = False
        self._emit_preview()

    def _on_sy_changed(self, v: float) -> None:
        if self._blocking: return
        self._blocking = True
        self._spin_h_pct.setValue(v * 100)
        if self._chk_aspect.isChecked():
            self._spin_sx.setValue(v)
            self._spin_w_pct.setValue(v * 100)
        self._blocking = False
        self._emit_preview()

    def _on_wpct_changed(self, v: float) -> None:
        if self._blocking: return
        self._blocking = True
        self._spin_sx.setValue(v / 100.0)
        if self._chk_aspect.isChecked():
            self._spin_sy.setValue(v / 100.0)
            self._spin_h_pct.setValue(v)
        self._blocking = False
        self._emit_preview()

    def _on_hpct_changed(self, v: float) -> None:
        if self._blocking: return
        self._blocking = True
        self._spin_sy.setValue(v / 100.0)
        if self._chk_aspect.isChecked():
            self._spin_sx.setValue(v / 100.0)
            self._spin_w_pct.setValue(v)
        self._blocking = False
        self._emit_preview()

    # ── Slot ─────────────────────────────────────────────────────────

    def _emit_preview(self, *_) -> None:
        self.preview_requested.emit(self._current_params())

    def _on_apply(self) -> None:
        self.apply_requested.emit(self._current_params())

    def _on_reset(self) -> None:
        self._blocking = True
        self._spin_sx.setValue(1.0); self._spin_sy.setValue(1.0)
        self._spin_w_pct.setValue(100); self._spin_h_pct.setValue(100)
        self._spin_angle.setValue(0.0)
        self._spin_skx.setValue(0.0); self._spin_sky.setValue(0.0)
        self._blocking = False
        self.reset_requested.emit()
