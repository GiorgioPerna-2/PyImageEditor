from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QDoubleSpinBox, QStackedWidget, QSizePolicy,
)


class TransformPanel(QWidget):
    """
    Pannello ritocchi in stile Windows Foto:
    - UI principale: pulsanti Rotazione, Ribaltamento e un grande bottone Ritaglia
    - Quando si entra in modalità crop, l'UI si trasforma mostrando
      istruzioni + Conferma / Annulla
    """

    # Segnali verso main_window
    rotate_requested = pyqtSignal(int)     # gradi esatti: -90, 90, 180
    flip_requested   = pyqtSignal(bool)    # True = orizzontale
    crop_start       = pyqtSignal()        # entra in modalità crop
    crop_confirm     = pyqtSignal()        # conferma crop
    crop_cancel      = pyqtSignal()        # annulla crop

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Usiamo QStackedWidget per alternare tra la UI normale e quella di crop
        self._stack = QStackedWidget()
        layout.addWidget(self._stack)

        self._stack.addWidget(self._build_normal_ui())   # page 0
        self._stack.addWidget(self._build_crop_ui())     # page 1
        self._stack.setCurrentIndex(0)

    # ------------------------------------------------------------------
    # Costruzione pagine
    # ------------------------------------------------------------------

    def _build_normal_ui(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        # ── Ritaglia ─────────────────────────────────────────────────
        btn_crop = QPushButton("✂  Ritaglia")
        btn_crop.setFixedHeight(42)
        btn_crop.setStyleSheet("font-size: 14px; font-weight: bold;")
        btn_crop.setToolTip("Clicca per disegnare l'area di ritaglio sull'immagine")
        btn_crop.clicked.connect(self._on_start_crop)
        layout.addWidget(btn_crop)

        # ── Ruota ─────────────────────────────────────────────────────
        rot_group = QGroupBox("Ruota")
        rot_layout = QHBoxLayout(rot_group)
        for label, deg, tip in [
            ("↺  90°",  -90, "Ruota 90° a sinistra"),
            ("↻  90°",   90, "Ruota 90° a destra"),
            ("180°",    180, "Ruota 180°"),
        ]:
            btn = QPushButton(label)
            btn.setToolTip(tip)
            btn.setFixedHeight(36)
            btn.clicked.connect(lambda _, d=deg: self.rotate_requested.emit(d))
            rot_layout.addWidget(btn)
        layout.addWidget(rot_group)

        # ── Ribalta ───────────────────────────────────────────────────
        flip_group = QGroupBox("Ribalta")
        flip_layout = QHBoxLayout(flip_group)
        btn_h = QPushButton("↔  Orizzontale")
        btn_v = QPushButton("↕  Verticale")
        btn_h.setFixedHeight(36)
        btn_v.setFixedHeight(36)
        btn_h.clicked.connect(lambda: self.flip_requested.emit(True))
        btn_v.clicked.connect(lambda: self.flip_requested.emit(False))
        flip_layout.addWidget(btn_h)
        flip_layout.addWidget(btn_v)
        layout.addWidget(flip_group)

        layout.addStretch()
        return w

    def _build_crop_ui(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        # Istruzioni
        lbl = QLabel(
            "<b>Modalità ritaglia</b><br><br>"
            "Trascina sull'immagine per selezionare l'area.<br><br>"
            "<i>Premi Invio per confermare, Esc per annullare.</i>"
        )
        lbl.setWordWrap(True)
        lbl.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.addWidget(lbl)

        btn_confirm = QPushButton("✓  Conferma")
        btn_confirm.setFixedHeight(40)
        btn_confirm.setStyleSheet("font-weight: bold; background: #2d7d46; color: white;")
        btn_confirm.clicked.connect(self._on_confirm_crop)
        layout.addWidget(btn_confirm)

        btn_cancel = QPushButton("✕  Annulla")
        btn_cancel.setFixedHeight(40)
        btn_cancel.setStyleSheet("font-weight: bold;")
        btn_cancel.clicked.connect(self._on_cancel_crop)
        layout.addWidget(btn_cancel)

        layout.addStretch()
        return w

    # ------------------------------------------------------------------
    # Slot
    # ------------------------------------------------------------------

    def _on_start_crop(self) -> None:
        self._stack.setCurrentIndex(1)
        self.crop_start.emit()

    def _on_confirm_crop(self) -> None:
        self._stack.setCurrentIndex(0)
        self.crop_confirm.emit()

    def _on_cancel_crop(self) -> None:
        self._stack.setCurrentIndex(0)
        self.crop_cancel.emit()

    def exit_crop_mode(self) -> None:
        """Chiamato dal main_window se il crop viene annullato esternamente (es. Esc)."""
        self._stack.setCurrentIndex(0)
