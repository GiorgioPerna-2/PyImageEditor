"""
ui/matting_panel.py
-------------------
Pannello scontornamento con due modalità:
  1. Automatica (rembg) — un click
  2. Interattiva (GrabCut) — l'utente disegna un rettangolo sul canvas

Il pannello non esegue direttamente le operazioni: emette segnali
che il main_window gestisce in thread separati.
"""
from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QComboBox, QCheckBox, QSlider, QSpinBox,
    QScrollArea, QFrame,
)


REMBG_MODELS = [
    ("u2net",               "U2Net — generale (default)"),
    ("u2net_human_seg",     "U2Net Human — ritratti"),
    ("isnet-general-use",   "ISNet — alta qualità (lento)"),
    ("silueta",             "Silueta — veloce, ritratti"),
]


class MattingPanel(QWidget):
    """Pannello con le due modalità di scontornamento."""

    # Emessi verso main_window
    rembg_requested      = pyqtSignal(str, bool, int, int, int)
    # model_name, alpha_matting, fg_thresh, bg_thresh, erode_size

    grabcut_start        = pyqtSignal()   # entra in modalità selezione rettangolo
    grabcut_cancel       = pyqtSignal()   # esce senza fare nulla
    grabcut_apply        = pyqtSignal(int)  # iterations
    # (il rettangolo viene letto direttamente dal canvas)

    refine_requested     = pyqtSignal(int, int, int)
    # erode_px, dilate_px, blur_px

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
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(10)
        scroll.setWidget(content)

        # ── Sezione automatica (rembg) ────────────────────────────────
        auto_group = QGroupBox("Automatico (rembg)")
        auto_layout = QVBoxLayout(auto_group)

        # Info / avviso
        info = QLabel(
            "⬇ Il modello viene scaricato (~170 MB) al primo utilizzo.\n"
            "Successivamente funziona offline."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #aaa; font-size: 10px;")
        auto_layout.addWidget(info)

        # Selezione modello
        auto_layout.addWidget(QLabel("Modello:"))
        self._combo_model = QComboBox()
        for key, label in REMBG_MODELS:
            self._combo_model.addItem(label, userData=key)
        auto_layout.addWidget(self._combo_model)

        # Alpha matting
        self._chk_alpha = QCheckBox("Alpha matting (bordi morbidi — capelli, pellicce)")
        self._chk_alpha.setToolTip(
            "Usa alpha matting per transizioni più naturali sui bordi fini.\n"
            "Richiede più tempo di elaborazione.")
        self._chk_alpha.toggled.connect(self._on_alpha_matting_toggled)
        auto_layout.addWidget(self._chk_alpha)

        # Parametri alpha matting (nascosti di default)
        self._alpha_params_widget = QWidget()
        ap_layout = QVBoxLayout(self._alpha_params_widget)
        ap_layout.setContentsMargins(12, 0, 0, 0)

        self._spin_fg = self._spinbox(240, 0, 255, "Soglia foreground:")
        self._spin_bg = self._spinbox(10, 0, 255, "Soglia background:")
        self._spin_erode = self._spinbox(10, 0, 40, "Erode size:")
        for label, spin in (
            ("Soglia foreground:", self._spin_fg),
            ("Soglia background:", self._spin_bg),
            ("Erode size:", self._spin_erode),
        ):
            row = QHBoxLayout()
            lbl = QLabel(label); lbl.setMinimumWidth(120)
            row.addWidget(lbl); row.addWidget(spin)
            ap_layout.addLayout(row)

        self._alpha_params_widget.setVisible(False)
        auto_layout.addWidget(self._alpha_params_widget)

        btn_auto = QPushButton("▶  Rimuovi sfondo automaticamente")
        btn_auto.setFixedHeight(38)
        btn_auto.setStyleSheet("font-weight: bold;")
        btn_auto.clicked.connect(self._on_rembg_clicked)
        auto_layout.addWidget(btn_auto)

        root.addWidget(auto_group)

        # ── Sezione interattiva (GrabCut) ─────────────────────────────
        gc_group = QGroupBox("Interattivo (GrabCut)")
        gc_layout = QVBoxLayout(gc_group)

        gc_info = QLabel(
            "Disegna un rettangolo intorno al soggetto da tenere.\n"
            "GrabCut usa OpenCV — nessun download richiesto."
        )
        gc_info.setWordWrap(True)
        gc_info.setStyleSheet("color: #aaa; font-size: 10px;")
        gc_layout.addWidget(gc_info)

        iter_row = QHBoxLayout()
        iter_row.addWidget(QLabel("Iterazioni:"))
        self._spin_iter = QSpinBox()
        self._spin_iter.setRange(1, 20)
        self._spin_iter.setValue(5)
        self._spin_iter.setToolTip("Più iterazioni = qualità migliore ma più lento")
        iter_row.addWidget(self._spin_iter)
        gc_layout.addLayout(iter_row)

        # Stack normale / modalità crop
        self._gc_normal_widget = QWidget()
        gc_normal_layout = QVBoxLayout(self._gc_normal_widget)
        gc_normal_layout.setContentsMargins(0, 0, 0, 0)
        btn_gc_start = QPushButton("✏  Disegna selezione sul canvas")
        btn_gc_start.setFixedHeight(38)
        btn_gc_start.setStyleSheet("font-weight: bold;")
        btn_gc_start.clicked.connect(self._on_grabcut_start)
        gc_normal_layout.addWidget(btn_gc_start)
        gc_layout.addWidget(self._gc_normal_widget)

        self._gc_active_widget = QWidget()
        gc_active_layout = QVBoxLayout(self._gc_active_widget)
        gc_active_layout.setContentsMargins(0, 0, 0, 0)
        lbl_active = QLabel(
            "<b>Modalità selezione attiva</b><br>"
            "Trascina un rettangolo sull'immagine.<br>"
            "Invio per applicare, Esc per annullare."
        )
        lbl_active.setWordWrap(True)
        gc_active_layout.addWidget(lbl_active)
        btn_gc_apply = QPushButton("✓  Applica GrabCut")
        btn_gc_apply.setFixedHeight(38)
        btn_gc_apply.setStyleSheet("font-weight: bold; background: #2d7d46; color: white;")
        btn_gc_apply.clicked.connect(self._on_grabcut_apply)
        btn_gc_cancel = QPushButton("✕  Annulla")
        btn_gc_cancel.clicked.connect(self._on_grabcut_cancel)
        gc_active_layout.addWidget(btn_gc_apply)
        gc_active_layout.addWidget(btn_gc_cancel)
        self._gc_active_widget.setVisible(False)
        gc_layout.addWidget(self._gc_active_widget)

        root.addWidget(gc_group)

        # ── Rifinitura bordi ──────────────────────────────────────────
        refine_group = QGroupBox("Rifinitura bordi")
        refine_layout = QVBoxLayout(refine_group)

        refine_info = QLabel(
            "Post-processing sulla maschera alpha del layer attivo.\n"
            "Applica dopo rembg o GrabCut per bordi più puliti."
        )
        refine_info.setWordWrap(True)
        refine_info.setStyleSheet("color: #aaa; font-size: 10px;")
        refine_layout.addWidget(refine_info)

        self._spin_r_erode  = self._spinbox(1, 0, 20, "Erode (px):")
        self._spin_r_dilate = self._spinbox(2, 0, 20, "Dilate (px):")
        self._spin_r_blur   = self._spinbox(3, 0, 20, "Blur (px):")
        for label, spin in (
            ("Erode (px):", self._spin_r_erode),
            ("Dilate (px):", self._spin_r_dilate),
            ("Blur (px):", self._spin_r_blur),
        ):
            row = QHBoxLayout()
            lbl = QLabel(label); lbl.setMinimumWidth(100)
            row.addWidget(lbl); row.addWidget(spin)
            refine_layout.addLayout(row)

        btn_refine = QPushButton("⚙  Applica rifinitura")
        btn_refine.clicked.connect(self._on_refine_clicked)
        refine_layout.addWidget(btn_refine)
        root.addWidget(refine_group)

        root.addStretch()

    # ── API pubblica ──────────────────────────────────────────────────

    def set_grabcut_active(self, active: bool) -> None:
        """Commuta tra UI normale e UI selezione attiva."""
        self._gc_normal_widget.setVisible(not active)
        self._gc_active_widget.setVisible(active)

    # ── Slot ─────────────────────────────────────────────────────────

    def _on_alpha_matting_toggled(self, checked: bool) -> None:
        self._alpha_params_widget.setVisible(checked)

    def _on_rembg_clicked(self) -> None:
        self.rembg_requested.emit(
            self._combo_model.currentData(),
            self._chk_alpha.isChecked(),
            self._spin_fg.value(),
            self._spin_bg.value(),
            self._spin_erode.value(),
        )

    def _on_grabcut_start(self) -> None:
        self.set_grabcut_active(True)
        self.grabcut_start.emit()

    def _on_grabcut_apply(self) -> None:
        self.set_grabcut_active(False)
        self.grabcut_apply.emit(self._spin_iter.value())

    def _on_grabcut_cancel(self) -> None:
        self.set_grabcut_active(False)
        self.grabcut_cancel.emit()

    def _on_refine_clicked(self) -> None:
        self.refine_requested.emit(
            self._spin_r_erode.value(),
            self._spin_r_dilate.value(),
            self._spin_r_blur.value(),
        )

    # ── Helper ────────────────────────────────────────────────────────

    @staticmethod
    def _spinbox(value: int, min_: int, max_: int, _: str) -> QSpinBox:
        s = QSpinBox()
        s.setRange(min_, max_)
        s.setValue(value)
        return s
