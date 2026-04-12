"""
ui/history_panel.py
-------------------
Pannello storico visuale: mostra le thumbnail dei snapshot nella history.
Cliccando su uno stato ci si salta direttamente (jump-to-history).

Il pannello è opzionale: può essere mostrato/nascosto dall'utente tramite
Visualizza → Storico. Non genera thumbnail automaticamente durante il disegno
per evitare il costo prestazionale — le miniature vengono create solo quando
la history viene aggiornata (_commit o undo/redo).
"""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal, QSize
from PyQt6.QtGui import QImage, QPixmap, QIcon, QColor
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QListWidget, QListWidgetItem,
    QLabel, QHBoxLayout, QPushButton,
)

THUMB_W = 80
THUMB_H = 60


def _array_to_pixmap(array: np.ndarray) -> QPixmap:
    """Scala l'array RGBA a THUMB_W × THUMB_H e lo converte in QPixmap."""
    from PIL import Image
    h, w = array.shape[:2]
    scale = min(THUMB_W / max(w, 1), THUMB_H / max(h, 1))
    tw = max(1, int(w * scale))
    th = max(1, int(h * scale))
    pil = Image.fromarray(array, "RGBA").resize((tw, th), Image.LANCZOS)
    # Sfondo a scacchiera
    checker = np.zeros((THUMB_H, THUMB_W, 3), dtype=np.uint8)
    for y in range(THUMB_H):
        for x in range(THUMB_W):
            checker[y, x] = 200 if (x // 4 + y // 4) % 2 == 0 else 160
    from PIL import Image as _I
    bg = _I.fromarray(checker, "RGB")
    ox, oy = (THUMB_W - tw) // 2, (THUMB_H - th) // 2
    bg.paste(pil, (ox, oy), pil)
    data = np.asarray(bg, dtype=np.uint8)
    qimg = QImage(data.tobytes(), THUMB_W, THUMB_H, THUMB_W * 3,
                  QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)


class HistoryPanel(QWidget):
    """
    Pannello storico visuale.

    Segnali:
      jump_requested(index)  — l'utente vuole saltare allo stato index
                               (0 = primo push, len-1 = più recente)
    """
    jump_requested = pyqtSignal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._current_index: int = -1

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        hdr = QHBoxLayout()
        hdr.addWidget(QLabel("<b>Storico</b>"))
        hdr.addStretch()
        btn_clear = QPushButton("Pulisci")
        btn_clear.setFixedHeight(22)
        btn_clear.setToolTip("Cancella lo storico (non modificabile)")
        btn_clear.setEnabled(False)   # solo informativo
        hdr.addWidget(btn_clear)
        layout.addLayout(hdr)

        self._list = QListWidget()
        self._list.setIconSize(QSize(THUMB_W, THUMB_H))
        self._list.setSpacing(2)
        self._list.itemClicked.connect(self._on_item_clicked)
        layout.addWidget(self._list, 1)

        self._lbl_count = QLabel("0 stati")
        self._lbl_count.setStyleSheet("font-size: 10px; color: #888;")
        layout.addWidget(self._lbl_count)

    # ── API pubblica ──────────────────────────────────────────────────

    def set_history(self, composites: list[np.ndarray], current_index: int) -> None:
        """
        Aggiorna il pannello con le thumbnail dei compositi.
        composites: lista di array RGBA (uno per snapshot)
        current_index: indice dello stato corrente
        """
        self._current_index = current_index
        self._list.blockSignals(True)
        self._list.clear()

        for i, arr in enumerate(composites):
            px = _array_to_pixmap(arr)
            item = QListWidgetItem(QIcon(px), f"Stato {i + 1}")
            item.setData(Qt.ItemDataRole.UserRole, i)
            item.setSizeHint(QSize(THUMB_W + 70, THUMB_H + 8))
            if i == current_index:
                item.setBackground(QColor(80, 120, 200, 80))
                f = item.font(); f.setBold(True); item.setFont(f)
            self._list.addItem(item)

        self._list.blockSignals(False)
        if 0 <= current_index < self._list.count():
            self._list.setCurrentRow(current_index)
            self._list.scrollToItem(self._list.currentItem())

        self._lbl_count.setText(f"{len(composites)} stati  (corrente: {current_index + 1})")

    # ── Slot interni ──────────────────────────────────────────────────

    def _on_item_clicked(self, item: QListWidgetItem) -> None:
        idx = item.data(Qt.ItemDataRole.UserRole)
        if idx is not None and idx != self._current_index:
            self.jump_requested.emit(idx)
