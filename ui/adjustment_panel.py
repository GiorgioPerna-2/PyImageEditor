"""
ui/adjustment_panel.py
----------------------
Pannello per creare e modificare layer di regolazione non-distruttivi.
Ogni tipo di regolazione ha la propria sezione con i controlli specifici.
"""
from __future__ import annotations
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider,
    QPushButton, QGroupBox, QComboBox, QScrollArea, QDoubleSpinBox,
    QSpinBox, QTabWidget,
)


def _slider(lo: int, hi: int, val: int) -> QSlider:
    s = QSlider(Qt.Orientation.Horizontal)
    s.setRange(lo, hi); s.setValue(val); return s


def _row(layout, label: str, widget) -> None:
    r = QHBoxLayout()
    lbl = QLabel(label); lbl.setMinimumWidth(130)
    r.addWidget(lbl); r.addWidget(widget)
    layout.addLayout(r)


class AdjustmentPanel(QWidget):
    """
    Pannello con un QTabWidget — una tab per tipo di regolazione.
    Emette add_requested(kind, params) quando l'utente clicca "Aggiungi".
    Emette update_requested(params) quando l'utente modifica i parametri
    (usato quando un AdjustmentLayer è selezionato nel layer panel).
    """

    add_requested    = pyqtSignal(str, dict)   # kind, params
    update_requested = pyqtSignal(dict)        # params aggiornati

    def __init__(self) -> None:
        super().__init__()
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

        info = QLabel(
            "Aggiungi un layer di regolazione sopra il layer attivo.\n"
            "Non modifica i pixel — si applica al volo durante il compositing."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #aaa; font-size: 10px;")
        root.addWidget(info)

        tabs = QTabWidget()
        root.addWidget(tabs)

        # ── Livelli ───────────────────────────────────────────────────
        lev_w, self._lev = self._build_levels()
        tabs.addTab(lev_w, "Livelli")

        # ── HSL ───────────────────────────────────────────────────────
        hsl_w, self._hsl = self._build_hsl()
        tabs.addTab(hsl_w, "HSL")

        # ── Bilanciamento colore ───────────────────────────────────────
        cb_w, self._cb = self._build_color_balance()
        tabs.addTab(cb_w, "Bilanciamento")

        # ── Curves ───────────────────────────────────────────────────
        cur_w, self._cur = self._build_curves()
        tabs.addTab(cur_w, "Curve")

        # ── Esposizione ───────────────────────────────────────────────
        exp_w, self._exp = self._build_exposure()
        tabs.addTab(exp_w, "Esposizione")

        self._tabs = tabs

        # Pulsanti azioni
        btn_row = QHBoxLayout()
        self._btn_add = QPushButton("＋ Aggiungi layer di regolazione")
        self._btn_add.setStyleSheet("font-weight: bold; padding: 5px;")
        self._btn_add.clicked.connect(self._on_add)
        btn_row.addWidget(self._btn_add)
        root.addLayout(btn_row)

        self._btn_update = QPushButton("↺ Aggiorna layer selezionato")
        self._btn_update.clicked.connect(self._on_update)
        self._btn_update.setVisible(False)
        root.addWidget(self._btn_update)

        root.addStretch()

    # ── Costruttori sezioni ───────────────────────────────────────────

    def _build_levels(self):
        w = QWidget(); lay = QVBoxLayout(w); lay.setSpacing(4)
        widgets = {}
        for key, label, lo, hi, val in [
            ("in_black",  "Input nero",   0, 255, 0),
            ("in_white",  "Input bianco", 0, 255, 255),
            ("out_black", "Output nero",  0, 255, 0),
            ("out_white", "Output bianco",0, 255, 255),
        ]:
            s = _slider(lo, hi, val)
            lbl_val = QLabel(str(val)); lbl_val.setMinimumWidth(30)
            s.valueChanged.connect(lambda v, l=lbl_val: l.setText(str(v)))
            r = QHBoxLayout()
            lbl = QLabel(label); lbl.setMinimumWidth(110)
            r.addWidget(lbl); r.addWidget(s); r.addWidget(lbl_val)
            lay.addLayout(r)
            widgets[key] = s
        # Gamma con spinbox
        gamma_spin = QDoubleSpinBox()
        gamma_spin.setRange(0.1, 10.0); gamma_spin.setSingleStep(0.1)
        gamma_spin.setValue(1.0); gamma_spin.setDecimals(2)
        _row(lay, "Gamma", gamma_spin)
        widgets["gamma_spin"] = gamma_spin
        return w, widgets

    def _build_hsl(self):
        w = QWidget(); lay = QVBoxLayout(w); lay.setSpacing(4)
        widgets = {}
        for key, label, lo, hi, val in [
            ("hue",        "Tonalità",    -180, 180, 0),
            ("saturation", "Saturazione", -100, 100, 0),
            ("lightness",  "Luminosità",  -100, 100, 0),
        ]:
            s = _slider(lo, hi, val)
            lbl_val = QLabel(str(val)); lbl_val.setMinimumWidth(35)
            s.valueChanged.connect(lambda v, l=lbl_val: l.setText(str(v)))
            r = QHBoxLayout()
            lbl = QLabel(label); lbl.setMinimumWidth(110)
            r.addWidget(lbl); r.addWidget(s); r.addWidget(lbl_val)
            lay.addLayout(r)
            widgets[key] = s
        return w, widgets

    def _build_color_balance(self):
        w = QWidget(); lay = QVBoxLayout(w); lay.setSpacing(4)
        widgets = {}
        for band in ("shadows", "midtones", "highlights"):
            label_band = {"shadows":"Ombre","midtones":"Mezzitoni","highlights":"Luci"}[band]
            gb = QGroupBox(label_band); gb_lay = QVBoxLayout(gb)
            for axis, label in [
                ("cyan_red",      "Ciano ↔ Rosso"),
                ("magenta_green", "Magenta ↔ Verde"),
                ("yellow_blue",   "Giallo ↔ Blu"),
            ]:
                key = f"{band}_{axis}"
                s = _slider(-100, 100, 0)
                lbl_val = QLabel("0"); lbl_val.setMinimumWidth(30)
                s.valueChanged.connect(lambda v, l=lbl_val: l.setText(str(v)))
                r = QHBoxLayout()
                lbl = QLabel(label); lbl.setMinimumWidth(130)
                r.addWidget(lbl); r.addWidget(s); r.addWidget(lbl_val)
                gb_lay.addLayout(r)
                widgets[key] = s
            lay.addWidget(gb)
        return w, widgets

    def _build_curves(self):
        w = QWidget(); lay = QVBoxLayout(w); lay.setSpacing(4)
        widgets = {}
        for ch, label in [("master","Master"),("r","Rosso"),("g","Verde"),("b","Blu")]:
            gb = QGroupBox(label); gb_lay = QVBoxLayout(gb)
            for pt, pt_label in [("in","Punto medio (input)"),("out","Punto medio (output)")]:
                key = f"{ch}_{pt}"
                s = _slider(0, 255, 128)
                lbl_val = QLabel("128"); lbl_val.setMinimumWidth(30)
                s.valueChanged.connect(lambda v, l=lbl_val: l.setText(str(v)))
                r = QHBoxLayout()
                lbl = QLabel(pt_label); lbl.setMinimumWidth(150)
                r.addWidget(lbl); r.addWidget(s); r.addWidget(lbl_val)
                gb_lay.addLayout(r)
                widgets[key] = s
            lay.addWidget(gb)
        return w, widgets

    def _build_exposure(self):
        w = QWidget(); lay = QVBoxLayout(w); lay.setSpacing(4)
        widgets = {}
        exp_spin = QDoubleSpinBox()
        exp_spin.setRange(-5.0, 5.0); exp_spin.setSingleStep(0.1)
        exp_spin.setValue(0.0); exp_spin.setDecimals(2); exp_spin.setSuffix(" EV")
        _row(lay, "Esposizione", exp_spin); widgets["exposure"] = exp_spin

        off_spin = QDoubleSpinBox()
        off_spin.setRange(-0.5, 0.5); off_spin.setSingleStep(0.01)
        off_spin.setValue(0.0); off_spin.setDecimals(3)
        _row(lay, "Offset", off_spin); widgets["offset"] = off_spin

        gam_spin = QDoubleSpinBox()
        gam_spin.setRange(0.1, 10.0); gam_spin.setSingleStep(0.1)
        gam_spin.setValue(1.0); gam_spin.setDecimals(2)
        _row(lay, "Gamma", gam_spin); widgets["gamma"] = gam_spin

        return w, widgets

    # ── Lettura parametri ─────────────────────────────────────────────

    def _current_kind_and_params(self) -> tuple[str, dict]:
        idx = self._tabs.currentIndex()
        kinds = ["levels", "hsl", "color_balance", "curves", "exposure"]
        kind = kinds[idx]
        params = self._read_params(kind)
        return kind, params

    def _read_params(self, kind: str) -> dict:
        if kind == "levels":
            w = self._lev
            return {
                "in_black":  w["in_black"].value(),
                "in_white":  w["in_white"].value(),
                "out_black": w["out_black"].value(),
                "out_white": w["out_white"].value(),
                "gamma":     w["gamma_spin"].value(),
            }
        if kind == "hsl":
            w = self._hsl
            return {k: w[k].value() for k in ("hue","saturation","lightness")}
        if kind == "color_balance":
            w = self._cb
            return {k: w[k].value() for k in w}
        if kind == "curves":
            w = self._cur
            return {k: w[k].value() for k in w}
        if kind == "exposure":
            w = self._exp
            return {k: w[k].value() for k in w}
        return {}

    # ── API pubblica ──────────────────────────────────────────────────

    def load_adjustment(self, kind: str, params: dict) -> None:
        """Carica i parametri di un AdjustmentLayer esistente nel pannello."""
        kind_to_idx = {"levels":0,"hsl":1,"color_balance":2,"curves":3,"exposure":4}
        idx = kind_to_idx.get(kind, 0)
        self._tabs.setCurrentIndex(idx)
        self._populate_params(kind, params)
        self._btn_update.setVisible(True)

    def _populate_params(self, kind: str, params: dict) -> None:
        if kind == "levels":
            w = self._lev
            for k in ("in_black","in_white","out_black","out_white"):
                if k in params: w[k].setValue(int(params[k]))
            if "gamma" in params: w["gamma_spin"].setValue(float(params["gamma"]))
        elif kind == "hsl":
            for k in ("hue","saturation","lightness"):
                if k in params: self._hsl[k].setValue(int(params[k]))
        elif kind == "color_balance":
            for k, s in self._cb.items():
                if k in params: s.setValue(int(params[k]))
        elif kind == "curves":
            for k, s in self._cur.items():
                if k in params: s.setValue(int(params[k]))
        elif kind == "exposure":
            for k, s in self._exp.items():
                if k in params: s.setValue(float(params[k]))

    def reset_ui(self) -> None:
        """Torna alla modalità 'aggiungi nuovo'."""
        self._btn_update.setVisible(False)

    # ── Slot ─────────────────────────────────────────────────────────

    def _on_add(self) -> None:
        kind, params = self._current_kind_and_params()
        self.add_requested.emit(kind, params)

    def _on_update(self) -> None:
        kind, params = self._current_kind_and_params()
        self.update_requested.emit(params)
