"""
ui/upscale_panel.py
-------------------
Pannello upscaling ESRGAN con selezione modello e pulsante di avvio.
L'elaborazione viene eseguita in un QThread separato.
"""
from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal, QThread, QObject
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QComboBox, QCheckBox, QGroupBox, QProgressBar,
)
import numpy as np
from core.upscale import available_models, model_is_cached


class _UpscaleWorker(QObject):
    progress = pyqtSignal(str)
    done     = pyqtSignal(object)   # np.ndarray
    error    = pyqtSignal(str)

    def __init__(self, rgba: np.ndarray, model_name: str) -> None:
        super().__init__()
        self._rgba = rgba
        self._model = model_name

    def run(self) -> None:
        try:
            from core.upscale import upscale
            result = upscale(self._rgba, self._model,
                             progress_cb=self.progress.emit)
            self.done.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))


class UpscalePanel(QWidget):
    upscale_requested = pyqtSignal(str)   # model_name — gestito da main_window

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Info
        info = QLabel(
            "Upscaling con modelli ESRGAN pre-trainati.\n"
            "Il modello viene scaricato la prima volta (~64 MB)\n"
            "e poi usato offline dalla cache locale."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #aaa; font-size: 10px;")
        layout.addWidget(info)

        # Selezione modello
        grp = QGroupBox("Modello")
        gl = QVBoxLayout(grp)
        self._combo = QComboBox()
        for m in available_models():
            cached = "✓" if model_is_cached(m) else "↓"
            self._combo.addItem(f"{cached} {m}", userData=m)
        gl.addWidget(self._combo)

        # Nota bilanciamento del bianco
        note = QLabel("✓ = in cache  ↓ = da scaricare")
        note.setStyleSheet("font-size: 10px; color: #888;")
        gl.addWidget(note)
        layout.addWidget(grp)

        # Pulsante
        self._btn = QPushButton("▶  Esegui upscaling")
        self._btn.setFixedHeight(36)
        self._btn.setStyleSheet("font-weight: bold;")
        self._btn.clicked.connect(self._on_click)
        layout.addWidget(self._btn)

        # Stato
        self._lbl_status = QLabel("")
        self._lbl_status.setWordWrap(True)
        self._lbl_status.setStyleSheet("font-size: 10px; color: #aaa;")
        layout.addWidget(self._lbl_status)

        layout.addStretch()

        self._thread: QThread | None = None
        self._worker: _UpscaleWorker | None = None

    @property
    def selected_model(self) -> str:
        return self._combo.currentData()

    def set_status(self, msg: str) -> None:
        self._lbl_status.setText(msg)

    def set_busy(self, busy: bool) -> None:
        self._btn.setEnabled(not busy)
        self._combo.setEnabled(not busy)

    def _on_click(self) -> None:
        self.upscale_requested.emit(self.selected_model)

    def refresh_cache_status(self) -> None:
        """Aggiorna le etichette ✓/↓ dopo un download."""
        current_data = self._combo.currentData()
        self._combo.blockSignals(True)
        self._combo.clear()
        for m in available_models():
            cached = "✓" if model_is_cached(m) else "↓"
            self._combo.addItem(f"{cached} {m}", userData=m)
        # Ripristina selezione
        for i in range(self._combo.count()):
            if self._combo.itemData(i) == current_data:
                self._combo.setCurrentIndex(i)
                break
        self._combo.blockSignals(False)
