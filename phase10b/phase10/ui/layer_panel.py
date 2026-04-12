from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal, QSize
from PyQt6.QtGui import QPixmap, QImage, QIcon, QFont, QColor
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTreeWidget, QTreeWidgetItem,
    QPushButton, QSlider, QLabel, QComboBox, QInputDialog,
    QFrame, QMenu, QToolButton,
)
import numpy as np
from core.layer import Layer, LayerGroup, BLEND_MODES
from core.layer_stack import LayerStack, Path

THUMB_W = 44
THUMB_H = 33


# ── Thumbnail helpers ─────────────────────────────────────────────────────────

def _make_thumb(array: np.ndarray) -> QPixmap:
    from PIL import Image
    h, w = array.shape[:2]
    scale = min(THUMB_W / w, THUMB_H / h)
    tw, th = max(1, int(w * scale)), max(1, int(h * scale))

    checker = np.zeros((THUMB_H, THUMB_W, 3), dtype=np.uint8)
    for y in range(THUMB_H):
        for x in range(THUMB_W):
            checker[y, x] = 200 if (x // 4 + y // 4) % 2 == 0 else 160

    pil = Image.fromarray(array, "RGBA").resize((tw, th), Image.LANCZOS)
    arr = np.asarray(pil, dtype=np.uint8)
    bg = Image.fromarray(checker, "RGB")
    fg = Image.fromarray(arr, "RGBA")
    bg.paste(fg, ((THUMB_W - tw) // 2, (THUMB_H - th) // 2), fg)
    final = np.asarray(bg, dtype=np.uint8)
    qimg = QImage(final.tobytes(), THUMB_W, THUMB_H, THUMB_W * 3,
                  QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)


def _make_group_icon() -> QPixmap:
    """Icona per i gruppi: cartella gialla."""
    px = QPixmap(THUMB_W, THUMB_H)
    px.fill(QColor(0, 0, 0, 0))
    from PyQt6.QtGui import QPainter, QPen, QBrush
    p = QPainter(px)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    # Corpo cartella
    p.setBrush(QBrush(QColor(240, 190, 40, 220)))
    p.setPen(QPen(QColor(180, 140, 20), 1))
    p.drawRoundedRect(2, 8, THUMB_W - 4, THUMB_H - 11, 3, 3)
    # Linguetta in alto
    p.drawRoundedRect(2, 4, 16, 8, 2, 2)
    p.end()
    return px


def _make_mask_thumb(mask: np.ndarray) -> QPixmap:
    from PIL import Image
    h, w = mask.shape
    scale = min((THUMB_W // 2) / w, THUMB_H / h)
    tw, th = max(1, int(w * scale)), max(1, int(h * scale))
    pil = Image.fromarray(mask, "L").resize((tw, th), Image.LANCZOS).convert("RGB")
    arr = np.asarray(pil, dtype=np.uint8)
    canvas = np.full((THUMB_H, THUMB_W // 2, 3), 128, dtype=np.uint8)
    oy = (THUMB_H - th) // 2; ox = ((THUMB_W // 2) - tw) // 2
    canvas[oy:oy+th, ox:ox+tw] = arr
    qimg = QImage(canvas.tobytes(), THUMB_W // 2, THUMB_H,
                  (THUMB_W // 2) * 3, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)


# ── Colonne QTreeWidget ───────────────────────────────────────────────────────
COL_VIS  = 0   # occhio
COL_ICON = 1   # thumbnail
COL_NAME = 2   # nome

_GROUP_ICON = None   # inizializzato lazy


class LayerPanel(QWidget):

    active_changed       = pyqtSignal(object)   # Path (tuple)
    visibility_toggled   = pyqtSignal(object)   # Path
    opacity_changed      = pyqtSignal(object, float)  # Path, opacity
    blend_changed        = pyqtSignal(object, str)    # Path, mode
    add_requested        = pyqtSignal()
    delete_requested     = pyqtSignal(object)   # Path
    duplicate_requested  = pyqtSignal(object)   # Path
    move_up_requested    = pyqtSignal(object)   # Path
    move_down_requested  = pyqtSignal(object)   # Path
    rename_requested     = pyqtSignal(object, str)    # Path, name
    import_as_layer      = pyqtSignal()

    # Gruppo
    create_group_requested  = pyqtSignal()
    group_node_requested    = pyqtSignal(object)   # Path da raggruppare
    ungroup_requested       = pyqtSignal(object)   # Path gruppo
    move_into_group_req     = pyqtSignal(object, object)  # node_path, group_path
    move_out_of_group_req   = pyqtSignal(object)   # node_path

    # Maschera
    mask_add_white  = pyqtSignal(object)
    mask_add_black  = pyqtSignal(object)
    mask_invert     = pyqtSignal(object)
    mask_apply      = pyqtSignal(object)
    mask_remove     = pyqtSignal(object)

    offset_reset    = pyqtSignal(object)

    def __init__(self) -> None:
        super().__init__()
        self._updating = False
        self._current_path: Path = (0,)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # ── Tree widget ───────────────────────────────────────────────
        self._tree = QTreeWidget()
        self._tree.setColumnCount(3)
        self._tree.setHeaderHidden(True)
        self._tree.setIconSize(QSize(THUMB_W, THUMB_H))
        self._tree.setColumnWidth(COL_VIS, 26)
        self._tree.setColumnWidth(COL_ICON, THUMB_W + 4)
        self._tree.setIndentation(16)
        self._tree.currentItemChanged.connect(self._on_current_changed)
        self._tree.itemDoubleClicked.connect(self._on_double_click)
        self._tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._tree.customContextMenuRequested.connect(self._on_context_menu)
        layout.addWidget(self._tree, 1)

        sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(sep)

        # ── Opacità ───────────────────────────────────────────────────
        op_row = QHBoxLayout()
        op_row.addWidget(QLabel("Opacità"))
        self._slider_opacity = QSlider(Qt.Orientation.Horizontal)
        self._slider_opacity.setRange(0, 100); self._slider_opacity.setValue(100)
        self._slider_opacity.valueChanged.connect(self._on_opacity_changed)
        self._lbl_opacity = QLabel("100%"); self._lbl_opacity.setMinimumWidth(36)
        op_row.addWidget(self._slider_opacity); op_row.addWidget(self._lbl_opacity)
        layout.addLayout(op_row)

        # ── Blend mode ────────────────────────────────────────────────
        self._combo_blend = QComboBox()
        for key, label in BLEND_MODES.items():
            self._combo_blend.addItem(label, userData=key)
        self._combo_blend.currentIndexChanged.connect(self._on_blend_changed)
        layout.addWidget(self._combo_blend)

        sep2 = QFrame(); sep2.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(sep2)

        # ── Pulsanti ─────────────────────────────────────────────────
        btn_row1 = QHBoxLayout()
        self._btn_add  = self._mkbtn("＋", "Nuovo layer trasparente")
        self._btn_del  = self._mkbtn("🗑", "Elimina")
        self._btn_dup  = self._mkbtn("⧉", "Duplica")
        self._btn_up   = self._mkbtn("↑",  "Sposta su")
        self._btn_down = self._mkbtn("↓",  "Sposta giù")
        for b in (self._btn_add, self._btn_del, self._btn_dup,
                  self._btn_up, self._btn_down):
            btn_row1.addWidget(b)
        layout.addLayout(btn_row1)

        btn_row2 = QHBoxLayout()
        btn_import = QPushButton("📂 Importa come layer")
        btn_import.clicked.connect(self.import_as_layer.emit)
        btn_row2.addWidget(btn_import)
        layout.addLayout(btn_row2)

        # Gruppo
        self._btn_group = QPushButton("📁 Gruppo ▾")
        group_menu = QMenu(self._btn_group)
        group_menu.addAction("Crea gruppo vuoto",
                             lambda: self.create_group_requested.emit())
        group_menu.addAction("Raggruppa nodo selezionato",
                             lambda: self.group_node_requested.emit(self._current_path))
        group_menu.addAction("Scioglie gruppo (ungroup)",
                             lambda: self.ungroup_requested.emit(self._current_path))
        group_menu.addSeparator()
        group_menu.addAction("Sposta fuori dal gruppo",
                             lambda: self.move_out_of_group_req.emit(self._current_path))
        self._btn_group.setMenu(group_menu)
        layout.addWidget(self._btn_group)

        # Maschera
        self._btn_mask = QPushButton("⬜ Maschera ▾")
        mask_menu = QMenu(self._btn_mask)
        mask_menu.addAction("Aggiungi maschera bianca",
                            lambda: self.mask_add_white.emit(self._current_path))
        mask_menu.addAction("Aggiungi maschera nera",
                            lambda: self.mask_add_black.emit(self._current_path))
        mask_menu.addSeparator()
        mask_menu.addAction("Inverti maschera",
                            lambda: self.mask_invert.emit(self._current_path))
        mask_menu.addAction("Applica maschera",
                            lambda: self.mask_apply.emit(self._current_path))
        mask_menu.addAction("Rimuovi maschera",
                            lambda: self.mask_remove.emit(self._current_path))
        self._btn_mask.setMenu(mask_menu)
        layout.addWidget(self._btn_mask)

        self._btn_pos = QPushButton("📐 Posizione ▾")
        pos_menu = QMenu(self._btn_pos)
        pos_menu.addAction("Centra layer nel canvas",
                           lambda: self.offset_reset.emit(self._current_path))
        self._btn_pos.setMenu(pos_menu)
        layout.addWidget(self._btn_pos)

        self._btn_add.clicked.connect(self.add_requested.emit)
        self._btn_del.clicked.connect(lambda: self.delete_requested.emit(self._current_path))
        self._btn_dup.clicked.connect(lambda: self.duplicate_requested.emit(self._current_path))
        self._btn_up.clicked.connect(lambda: self.move_up_requested.emit(self._current_path))
        self._btn_down.clicked.connect(lambda: self.move_down_requested.emit(self._current_path))

    # ── Refresh ───────────────────────────────────────────────────────

    def refresh_from_stack(self, stack: LayerStack) -> None:
        self._updating = True
        self._tree.blockSignals(True)
        self._tree.clear()
        self._current_path = stack.active_path

        # Popola ricorsivamente dall'alto (index alto = prima riga)
        self._populate(self._tree.invisibleRootItem(),
                       list(reversed(list(enumerate(stack.nodes)))),
                       prefix=(), stack_len=len(stack.nodes),
                       active_path=stack.active_path)

        # Espandi tutto
        self._tree.expandAll()

        # Seleziona l'item attivo
        self._select_item_by_path(stack.active_path)

        # Aggiorna slider e blend per il nodo attivo (o il suo padre gruppo)
        try:
            active_node = stack.get_node(stack.active_path)
            self._slider_opacity.blockSignals(True)
            self._slider_opacity.setValue(int(active_node.opacity * 100))
            self._slider_opacity.blockSignals(False)
            self._lbl_opacity.setText(f"{int(active_node.opacity * 100)}%")
            blend_keys = list(BLEND_MODES.keys())
            if active_node.blend_mode in blend_keys:
                self._combo_blend.blockSignals(True)
                self._combo_blend.setCurrentIndex(blend_keys.index(active_node.blend_mode))
                self._combo_blend.blockSignals(False)
        except Exception:
            pass

        self._tree.blockSignals(False)
        self._updating = False

    def _populate(self, parent_item, reversed_indexed_nodes, prefix: tuple,
                  stack_len: int, active_path: Path) -> None:
        """
        reversed_indexed_nodes: lista di (original_index, node) già invertita
        (così il layer più in alto appare primo nella lista).
        prefix: path del parent (vuoto per root).
        """
        for orig_idx, node in reversed_indexed_nodes:
            path = (*prefix, orig_idx)
            item = QTreeWidgetItem(parent_item)
            item.setData(COL_NAME, Qt.ItemDataRole.UserRole, path)

            if isinstance(node, LayerGroup):
                icon_px = _make_group_icon()
                item.setIcon(COL_ICON, QIcon(icon_px))
                vis_text = "👁" if node.visible else " "
                item.setText(COL_VIS, vis_text)
                item.setText(COL_NAME, f"📁 {node.name}")
                item.setForeground(COL_NAME, QColor(200, 160, 60))
                # Figli (invertiti)
                children_rev = list(reversed(list(enumerate(node.children))))
                self._populate(item, children_rev, prefix=path,
                               stack_len=stack_len, active_path=active_path)
            else:
                thumb = _make_thumb(node.data)
                item.setIcon(COL_ICON, QIcon(thumb))
                vis_text = "👁" if node.visible else " "
                item.setText(COL_VIS, vis_text)
                name = node.name
                if node.mask is not None:
                    name += " 🎭"
                item.setText(COL_NAME, name)
                if path == active_path:
                    f = item.font(COL_NAME); f.setBold(True); item.setFont(COL_NAME, f)
                    item.setBackground(COL_NAME, QColor(80, 120, 200, 60))

    def _select_item_by_path(self, path: Path) -> None:
        def search(parent):
            for i in range(parent.childCount()):
                child = parent.child(i)
                if child.data(COL_NAME, Qt.ItemDataRole.UserRole) == path:
                    self._tree.setCurrentItem(child)
                    return True
                if search(child):
                    return True
            return False
        search(self._tree.invisibleRootItem())

    # ── Slot interni ──────────────────────────────────────────────────

    def _on_current_changed(self, current, _prev) -> None:
        if self._updating or current is None:
            return
        path = current.data(COL_NAME, Qt.ItemDataRole.UserRole)
        if path is not None:
            self._current_path = path
            self.active_changed.emit(path)

    def _on_double_click(self, item: QTreeWidgetItem, _col: int) -> None:
        path = item.data(COL_NAME, Qt.ItemDataRole.UserRole)
        if path is None:
            return
        current_name = item.text(COL_NAME).replace("📁 ", "").replace(" 🎭", "").strip()
        name, ok = QInputDialog.getText(self, "Rinomina", "Nome:", text=current_name)
        if ok and name.strip():
            self.rename_requested.emit(path, name.strip())

    def _on_context_menu(self, pos) -> None:
        item = self._tree.itemAt(pos)
        if item is None:
            return
        path = item.data(COL_NAME, Qt.ItemDataRole.UserRole)
        if path is None:
            return

        menu = QMenu(self)
        menu.addAction("Rinomina", lambda: self._on_double_click(item, COL_NAME))
        menu.addSeparator()

        # Eye toggle
        is_visible = "👁" in item.text(COL_VIS)
        label_vis = "Nascondi" if is_visible else "Mostra"
        menu.addAction(label_vis, lambda: self.visibility_toggled.emit(path))

        menu.addSeparator()
        menu.addAction("Duplica", lambda: self.duplicate_requested.emit(path))
        menu.addAction("Elimina", lambda: self.delete_requested.emit(path))
        menu.addSeparator()

        # Operazioni gruppo
        node_text = item.text(COL_NAME)
        if "📁" in node_text:
            menu.addAction("Scioglie gruppo", lambda: self.ungroup_requested.emit(path))
        else:
            menu.addAction("Raggruppa questo layer",
                           lambda: self.group_node_requested.emit(path))

        if len(path) > 1:
            menu.addAction("Sposta fuori dal gruppo",
                           lambda: self.move_out_of_group_req.emit(path))

        # Sposta dentro un gruppo: lista gruppi disponibili
        groups = self._collect_groups()
        if groups:
            into_menu = menu.addMenu("Sposta dentro gruppo…")
            for gpath, gname in groups:
                if gpath != path and not (len(path) > len(gpath) and path[:len(gpath)] == gpath):
                    into_menu.addAction(gname,
                        lambda _p=path, _g=gpath: self.move_into_group_req.emit(_p, _g))

        menu.exec(self._tree.viewport().mapToGlobal(pos))

    def _collect_groups(self) -> list[tuple[Path, str]]:
        result: list[tuple[Path, str]] = []
        def visit(parent, prefix):
            for i in range(parent.childCount()):
                child = parent.child(i)
                path = child.data(COL_NAME, Qt.ItemDataRole.UserRole)
                if path and "📁" in child.text(COL_NAME):
                    result.append((path, child.text(COL_NAME)))
                visit(child, path)
        visit(self._tree.invisibleRootItem(), ())
        return result

    def _on_opacity_changed(self, value: int) -> None:
        self._lbl_opacity.setText(f"{value}%")
        if not self._updating:
            self.opacity_changed.emit(self._current_path, value / 100.0)

    def _on_blend_changed(self, _: int) -> None:
        if not self._updating:
            self.blend_changed.emit(self._current_path, self._combo_blend.currentData())

    @staticmethod
    def _mkbtn(text: str, tip: str) -> QPushButton:
        b = QPushButton(text); b.setToolTip(tip); b.setFixedWidth(34); return b
