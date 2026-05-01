"""ui/inpainting_panel.py"""
from __future__ import annotations
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QSpinBox, QGroupBox, QComboBox, QProgressBar,
)


class InpaintingPanel(QWidget):
    inpaint_requested  = pyqtSignal(str, int, int)   # (method, patch_size, radius)
    download_requested = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setSpacing(8)
        root.setContentsMargins(6, 6, 6, 6)

        # ── Stato modello LaMa ────────────────────────────────────────
        lbox = QGroupBox("Modello LaMa AI")
        ll = QVBoxLayout(lbox)
        self._lbl_lama = QLabel()
        self._lbl_lama.setWordWrap(True)
        ll.addWidget(self._lbl_lama)
        self._btn_download = QPushButton("⬇ Scarica modello (~210 MB)")
        self._btn_download.clicked.connect(self.download_requested)
        ll.addWidget(self._btn_download)
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setVisible(False)
        ll.addWidget(self._progress)
        self._lbl_progress = QLabel("")
        self._lbl_progress.setStyleSheet("color:#888; font-size:10px;")
        self._lbl_progress.setVisible(False)
        ll.addWidget(self._lbl_progress)
        root.addWidget(lbox)

        # ── Metodo ────────────────────────────────────────────────────
        mbox = QGroupBox("Metodo")
        ml = QVBoxLayout(mbox)
        self._method_combo = QComboBox()
        self._method_combo.addItem("LaMa AI (massima qualità)", "lama")
        self._method_combo.addItem("Exemplar-based (buona qualità)", "patch")
        self._method_combo.addItem("OpenCV Telea (veloce, aree piccole)", "opencv")
        self._method_combo.currentIndexChanged.connect(self._on_method_changed)
        ml.addWidget(self._method_combo)

        # patch size
        self._patch_row = QHBoxLayout()
        self._patch_row.addWidget(QLabel("Dimensione patch:"))
        self._spin_patch = QSpinBox()
        self._spin_patch.setRange(5, 21); self._spin_patch.setValue(9)
        self._spin_patch.setSingleStep(2); self._spin_patch.setSuffix(" px")
        self._patch_row.addWidget(self._spin_patch); self._patch_row.addStretch()
        ml.addLayout(self._patch_row)

        # radius
        self._radius_row = QHBoxLayout()
        self._radius_row.addWidget(QLabel("Raggio:"))
        self._spin_radius = QSpinBox()
        self._spin_radius.setRange(1, 50); self._spin_radius.setValue(10)
        self._spin_radius.setSuffix(" px")
        self._radius_row.addWidget(self._spin_radius); self._radius_row.addStretch()
        ml.addLayout(self._radius_row)
        root.addWidget(mbox)

        # ── Istruzioni ────────────────────────────────────────────────
        hint = QLabel(
            "<i>1. Seleziona l'area da rimuovere<br>"
            "   (lasso, rettangolo o bacchetta magica).<br>"
            "2. Clicca <b>Applica</b>.<br><br>"
            "LaMa: qualità AI, richiede download.<br>"
            "Exemplar: buona qualità, nessun download.<br>"
            "OpenCV: istantaneo, aree molto piccole.</i>")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#888; font-size:10px;")
        root.addWidget(hint)

        # ── Applica ───────────────────────────────────────────────────
        self._btn_apply = QPushButton("✨ Applica inpainting")
        self._btn_apply.setStyleSheet("font-weight:bold; padding:6px;")
        self._btn_apply.clicked.connect(self._on_apply)
        root.addWidget(self._btn_apply)
        root.addStretch()

        self._on_method_changed(0)
        self._refresh_lama_status()

    def _refresh_lama_status(self) -> None:
        from core.inpainting import is_lama_available
        if is_lama_available():
            self._lbl_lama.setText("✅ Modello LaMa pronto.")
            self._lbl_lama.setStyleSheet("color:#4a4;")
            self._btn_download.setVisible(False)
        else:
            self._lbl_lama.setText("⚠ Modello non scaricato.\nSenza LaMa viene usato Exemplar-based.")
            self._lbl_lama.setStyleSheet("color:#a80;")
            self._btn_download.setVisible(True)

    def _on_method_changed(self, idx: int) -> None:
        key = self._method_combo.itemData(idx)
        for i in range(self._patch_row.count()):
            w = self._patch_row.itemAt(i).widget()
            if w: w.setVisible(key == "patch")
        for i in range(self._radius_row.count()):
            w = self._radius_row.itemAt(i).widget()
            if w: w.setVisible(key == "opencv")

    def _on_apply(self) -> None:
        self.inpaint_requested.emit(
            self._method_combo.currentData(),
            self._spin_patch.value(),
            self._spin_radius.value(),
        )

    # ── Callbacks da main_window ──────────────────────────────────────
    def on_download_started(self) -> None:
        self._btn_download.setEnabled(False)
        self._progress.setValue(0)
        self._progress.setVisible(True)
        self._lbl_progress.setVisible(True)

    def on_download_progress(self, fraction: float, message: str) -> None:
        self._progress.setValue(int(fraction * 100))
        self._lbl_progress.setText(message)

    def on_download_done(self) -> None:
        self._progress.setVisible(False)
        self._lbl_progress.setVisible(False)
        self._btn_download.setEnabled(True)
        self._refresh_lama_status()

    def on_download_error(self, msg: str) -> None:
        self._progress.setVisible(False)
        self._lbl_progress.setVisible(False)
        self._btn_download.setEnabled(True)
        self._lbl_lama.setText(f"❌ Download fallito: {msg}")
        self._lbl_lama.setStyleSheet("color:#c00;")

    def on_inpaint_started(self) -> None:
        self._btn_apply.setEnabled(False)
        self._btn_apply.setText("⏳ Elaborazione…")

    def on_inpaint_done(self) -> None:
        self._btn_apply.setEnabled(True)
        self._btn_apply.setText("✨ Applica inpainting")
