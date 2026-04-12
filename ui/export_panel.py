from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QGroupBox, QComboBox, QSlider,
    QCheckBox, QSpinBox,
)

# Formato → (etichetta, estensione, supporta qualità, supporta alpha)
FORMATS: dict[str, tuple[str, str, bool, bool]] = {
    "PNG":  ("PNG (lossless)",          ".png",  False, True),
    "JPEG": ("JPEG (lossy)",            ".jpg",  True,  False),
    "WebP": ("WebP (lossy/lossless)",   ".webp", True,  True),
    "TIFF": ("TIFF (lossless)",         ".tiff", False, True),
    "BMP":  ("BMP (no compressione)",   ".bmp",  False, True),
}


class ExportPanel(QWidget):
    """
    Pannello per esportare l'immagine in un formato diverso dall'originale,
    con opzioni di qualità e ridimensionamento.
    """

    export_requested = pyqtSignal(str, int, int)
    # str  = format key (es. "JPEG")
    # int  = quality 0-100
    # int  = resize % (100 = nessun ridimensionamento)

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # ── Formato ───────────────────────────────────────────────────
        fmt_group = QGroupBox("Formato")
        fmt_layout = QVBoxLayout(fmt_group)
        self._combo_fmt = QComboBox()
        for key, (label, ext, _, _) in FORMATS.items():
            self._combo_fmt.addItem(f"{label}  [{ext}]", userData=key)
        self._combo_fmt.currentIndexChanged.connect(self._on_format_changed)
        fmt_layout.addWidget(self._combo_fmt)

        self._lbl_alpha_warn = QLabel("⚠ JPEG non supporta la trasparenza (alpha verrà rimossa)")
        self._lbl_alpha_warn.setWordWrap(True)
        self._lbl_alpha_warn.setStyleSheet("color: orange; font-size: 10px;")
        self._lbl_alpha_warn.setVisible(False)
        fmt_layout.addWidget(self._lbl_alpha_warn)
        layout.addWidget(fmt_group)

        # ── Qualità (solo JPEG / WebP) ────────────────────────────────
        self._quality_group = QGroupBox("Qualità")
        q_layout = QVBoxLayout(self._quality_group)
        q_row = QHBoxLayout()
        self._slider_quality = QSlider(Qt.Orientation.Horizontal)
        self._slider_quality.setMinimum(1)
        self._slider_quality.setMaximum(100)
        self._slider_quality.setValue(92)
        self._lbl_quality_val = QLabel("92")
        self._lbl_quality_val.setMinimumWidth(28)
        self._slider_quality.valueChanged.connect(
            lambda v: self._lbl_quality_val.setText(str(v))
        )
        q_row.addWidget(self._slider_quality)
        q_row.addWidget(self._lbl_quality_val)
        q_layout.addLayout(q_row)
        layout.addWidget(self._quality_group)

        # ── Ridimensionamento ─────────────────────────────────────────
        resize_group = QGroupBox("Ridimensionamento")
        resize_layout = QVBoxLayout(resize_group)
        self._chk_resize = QCheckBox("Ridimensiona")
        resize_layout.addWidget(self._chk_resize)

        resize_row = QHBoxLayout()
        self._spin_resize = QSpinBox()
        self._spin_resize.setMinimum(1)
        self._spin_resize.setMaximum(400)
        self._spin_resize.setValue(100)
        self._spin_resize.setSuffix(" %")
        self._spin_resize.setEnabled(False)
        self._chk_resize.toggled.connect(self._spin_resize.setEnabled)
        resize_row.addWidget(QLabel("Scala:"))
        resize_row.addWidget(self._spin_resize)
        resize_layout.addLayout(resize_row)
        layout.addWidget(resize_group)

        # ── Pulsante Esporta ──────────────────────────────────────────
        btn_export = QPushButton("💾  Esporta…")
        btn_export.setStyleSheet("font-weight: bold; padding: 6px;")
        btn_export.clicked.connect(self._on_export)
        layout.addWidget(btn_export)

        layout.addStretch()

        # Aggiorna visibilità qualità in base al formato iniziale
        self._on_format_changed(0)

    # ------------------------------------------------------------------

    def _on_format_changed(self, _: int) -> None:
        key = self._combo_fmt.currentData()
        _, _, has_quality, has_alpha = FORMATS.get(key, ("", "", False, True))
        self._quality_group.setVisible(has_quality)
        self._lbl_alpha_warn.setVisible(not has_alpha)

    def _on_export(self) -> None:
        key = self._combo_fmt.currentData()
        quality = self._slider_quality.value()
        resize_pct = self._spin_resize.value() if self._chk_resize.isChecked() else 100
        self.export_requested.emit(key, quality, resize_pct)

    # ------------------------------------------------------------------

    @property
    def current_format(self) -> str:
        return self._combo_fmt.currentData()

    @property
    def current_extension(self) -> str:
        key = self.current_format
        return FORMATS[key][1]
