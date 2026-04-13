from __future__ import annotations

import numpy as np
from .layer import Layer, LayerGroup, AdjustmentLayer, NodeType, node_from_snapshot

# Tipo path: tupla di indici che naviga l'albero dall'alto
# Es: (2,) = nodo top-level index 2
#     (2, 1) = figlio index 1 del gruppo top-level index 2
Path = tuple[int, ...]


class LayerStack:
    """
    Albero di layer. Index 0 = bottom, index alto = primo piano (top).
    Le dimensioni del canvas sono memorizzate esplicitamente — nessun vincolo
    sulla posizione o sul tipo di nodes[0].

    _active_path punta sempre a un Layer foglia (mai a un gruppo).
    """

    def __init__(self, canvas_w: int = 0, canvas_h: int = 0) -> None:
        self._canvas_w = canvas_w
        self._canvas_h = canvas_h
        self._nodes: list[NodeType] = []
        self._active_path: Path = (0,)

    # ── Factory ───────────────────────────────────────────────────────

    @classmethod
    def from_image(cls, array: np.ndarray, name: str = "Sfondo") -> "LayerStack":
        h, w = array.shape[:2]
        stack = cls(canvas_w=w, canvas_h=h)
        stack._nodes = [Layer(array.copy(), name)]
        stack._active_path = (0,)
        return stack

    # ── Navigazione albero ────────────────────────────────────────────

    def get_node(self, path: Path) -> NodeType:
        nodes = self._nodes
        for depth, idx in enumerate(path):
            node = nodes[idx]
            if depth == len(path) - 1:
                return node
            assert isinstance(node, LayerGroup), f"Nodo a {path[:depth+1]} non è un gruppo"
            nodes = node.children
        raise IndexError(f"Path vuoto: {path}")

    def get_parent_list(self, path: Path) -> list[NodeType]:
        """Lista che contiene il nodo al path dato."""
        if len(path) == 1:
            return self._nodes
        parent = self.get_node(path[:-1])
        assert isinstance(parent, LayerGroup)
        return parent.children

    def _find_first_leaf(self, path: Path) -> Path | None:
        """Ritorna il path del primo Layer/AdjustmentLayer foglia dentro il nodo indicato."""
        node = self.get_node(path)
        if isinstance(node, (Layer, AdjustmentLayer)):
            return path
        if isinstance(node, LayerGroup):
            for i, child in enumerate(node.children):
                result = self._find_first_leaf((*path, i))
                if result is not None:
                    return result
        return None

    def _all_leaf_paths(self) -> list[Path]:
        """Tutti i path alle foglie Layer/AdjustmentLayer nell'albero, ordine di visita."""
        result: list[Path] = []
        self._collect_leaves(self._nodes, (), result)
        return result

    def _collect_leaves(self, nodes: list[NodeType], prefix: Path,
                        result: list[Path]) -> None:
        for i, node in enumerate(nodes):
            path = (*prefix, i)
            if isinstance(node, (Layer, AdjustmentLayer)):
                result.append(path)
            elif isinstance(node, LayerGroup):
                self._collect_leaves(node.children, path, result)

    # ── Proprietà ─────────────────────────────────────────────────────

    @property
    def active(self) -> Layer | AdjustmentLayer:
        node = self.get_node(self._active_path)
        assert isinstance(node, (Layer, AdjustmentLayer)), \
            "active_path punta a un gruppo"
        return node

    @property
    def active_path(self) -> Path:
        return self._active_path

    @property
    def nodes(self) -> list[NodeType]:
        return self._nodes

    @property
    def canvas_size(self) -> tuple[int, int]:
        """(width, height) del canvas."""
        return self._canvas_w, self._canvas_h

    def get_node_at_path(self, path: Path) -> NodeType:
        return self.get_node(path)

    # ── Selezione ─────────────────────────────────────────────────────

    def set_active_path(self, path: Path) -> None:
        node = self.get_node(path)
        if isinstance(node, (Layer, AdjustmentLayer)):
            self._active_path = path
        elif isinstance(node, LayerGroup):
            leaf = self._find_first_leaf(path)
            if leaf:
                self._active_path = leaf

    # ── Gestione nodi ─────────────────────────────────────────────────

    def add_layer(self, name: str | None = None) -> Path:
        """Inserisce un Layer trasparente sopra il nodo attivo."""
        if name is None:
            name = f"Layer {self._count_all() + 1}"
        parent_list = self.get_parent_list(self._active_path)
        insert_at = self._active_path[-1] + 1
        parent_list.insert(insert_at,
            Layer(np.zeros((self._canvas_h, self._canvas_w, 4), dtype=np.uint8), name))
        new_path = (*self._active_path[:-1], insert_at)
        self._active_path = new_path
        return new_path

    def add_layer_from_array(self, array: np.ndarray, name: str,
                             offset_x: int = 0, offset_y: int = 0) -> Path:
        parent_list = self.get_parent_list(self._active_path)
        insert_at = self._active_path[-1] + 1
        parent_list.insert(insert_at, Layer(array.copy(), name,
                                            offset_x=offset_x, offset_y=offset_y))
        new_path = (*self._active_path[:-1], insert_at)
        self._active_path = new_path
        return new_path

    def remove_node(self, path: Path) -> bool:
        if len(self._all_leaf_paths()) <= 1:
            return False   # deve restare almeno un Layer
        parent_list = self.get_parent_list(path)
        idx = path[-1]
        parent_list.pop(idx)
        # Aggiusta active_path se necessario
        self._fix_active_after_remove(path)
        return True

    def duplicate_node(self, path: Path) -> Path:
        node = self.get_node(path)
        clone = node.clone()
        if isinstance(clone, Layer):
            clone.name = clone.name + " copia"
        else:
            clone.name = clone.name + " copia"
        parent_list = self.get_parent_list(path)
        insert_at = path[-1] + 1
        parent_list.insert(insert_at, clone)
        new_path = (*path[:-1], insert_at)
        self._active_path = new_path if isinstance(clone, Layer) else \
            self._find_first_leaf(new_path) or self._active_path
        return new_path

    def move_up(self, path: Path) -> bool:
        parent_list = self.get_parent_list(path)
        idx = path[-1]
        if idx >= len(parent_list) - 1:
            return False
        parent_list[idx], parent_list[idx + 1] = parent_list[idx + 1], parent_list[idx]
        if self._active_path == path:
            self._active_path = (*path[:-1], idx + 1)
        elif self._active_path == (*path[:-1], idx + 1):
            self._active_path = path
        return True

    def move_down(self, path: Path) -> bool:
        parent_list = self.get_parent_list(path)
        idx = path[-1]
        if idx <= 0:
            return False
        parent_list[idx], parent_list[idx - 1] = parent_list[idx - 1], parent_list[idx]
        if self._active_path == path:
            self._active_path = (*path[:-1], idx - 1)
        elif self._active_path == (*path[:-1], idx - 1):
            self._active_path = path
        return True

    # ── Operazioni gruppo ─────────────────────────────────────────────

    def create_group(self, name: str = "Gruppo") -> Path:
        """Crea un gruppo vuoto sopra il nodo attivo."""
        parent_list = self.get_parent_list(self._active_path)
        insert_at = self._active_path[-1] + 1
        group = LayerGroup(name=name)
        parent_list.insert(insert_at, group)
        return (*self._active_path[:-1], insert_at)

    def group_node(self, path: Path, group_name: str = "Gruppo") -> Path:
        """Racchiude il nodo al path in un nuovo gruppo."""
        parent_list = self.get_parent_list(path)
        idx = path[-1]
        node = parent_list.pop(idx)
        group = LayerGroup(name=group_name, children=[node])
        parent_list.insert(idx, group)
        group_path = (*path[:-1], idx)
        # Active punta al figlio dentro il gruppo
        if isinstance(node, Layer):
            self._active_path = (*group_path, 0)
        else:
            leaf = self._find_first_leaf((*group_path, 0))
            self._active_path = leaf or self._active_path
        return group_path

    def ungroup(self, path: Path) -> bool:
        """Scioglie il gruppo: i figli vengono spostati nella posizione del gruppo."""
        node = self.get_node(path)
        if not isinstance(node, LayerGroup):
            return False
        parent_list = self.get_parent_list(path)
        idx = path[-1]
        parent_list.pop(idx)
        for i, child in enumerate(node.children):
            parent_list.insert(idx + i, child)
        # Aggiusta active
        all_leaves = self._all_leaf_paths()
        if all_leaves:
            # Prova a trovare un leaf vicino
            self._active_path = all_leaves[min(idx, len(all_leaves) - 1)]
        return True

    def move_into_group(self, node_path: Path, group_path: Path) -> bool:
        """Sposta il nodo node_path come figlio del gruppo group_path."""
        if node_path == group_path:
            return False
        # Verifica che group_path sia un gruppo
        dest = self.get_node(group_path)
        if not isinstance(dest, LayerGroup):
            return False
        # Non permettere di spostare un antenato dentro un suo discendente
        if len(group_path) >= len(node_path) and \
                group_path[:len(node_path)] == node_path:
            return False
        node = self.get_node(node_path)
        parent_list = self.get_parent_list(node_path)
        parent_list.pop(node_path[-1])
        dest.children.insert(len(dest.children), node)
        new_path = (*group_path, len(dest.children) - 1)
        if isinstance(node, Layer):
            self._active_path = new_path
        else:
            leaf = self._find_first_leaf(new_path)
            self._active_path = leaf or self._active_path
        return True

    def move_out_of_group(self, path: Path) -> bool:
        """Sposta il nodo fuori dal suo gruppo parent, dopo di esso."""
        if len(path) < 2:
            return False   # già al top level
        node = self.get_node(path)
        parent_list = self.get_parent_list(path)
        parent_list.pop(path[-1])
        # Inserisci dopo il gruppo parent nel livello superiore
        grandparent_list = self.get_parent_list(path[:-1]) if len(path) > 2 else self._nodes
        insert_at = path[-2] + 1
        grandparent_list.insert(insert_at, node)
        new_path = (*path[:-2], insert_at) if len(path) > 2 else (insert_at,)
        if isinstance(node, Layer):
            self._active_path = new_path
        else:
            leaf = self._find_first_leaf(new_path)
            self._active_path = leaf or self._active_path
        return True

    # ── Dati layer attivo ─────────────────────────────────────────────

    def set_active_data(self, array: np.ndarray) -> None:
        self.active.data = array.copy()

    def set_active_offset(self, dx: int, dy: int) -> None:
        layer = self.active
        layer.offset_x += dx
        layer.offset_y += dy

    def set_active_offset_absolute(self, x: int, y: int) -> None:
        layer = self.active
        layer.offset_x = x
        layer.offset_y = y

    # ── Compositing ───────────────────────────────────────────────────

    def composite(self) -> np.ndarray:
        result = self._composite_nodes(self._nodes, self._canvas_w, self._canvas_h)
        return np.clip(result, 0, 255).astype(np.uint8)

    def composite_with_override(self, path: Path, override_data: np.ndarray) -> np.ndarray:
        result = self._composite_nodes(self._nodes, self._canvas_w, self._canvas_h,
                                       override_path=path, override_data=override_data)
        return np.clip(result, 0, 255).astype(np.uint8)

    def composite_without_active(self) -> np.ndarray:
        """Composta tutti i layer ESCLUSO quello attivo. Usato come cache durante il disegno."""
        result = self._composite_nodes(self._nodes, self._canvas_w, self._canvas_h,
                                       skip_path=self._active_path)
        return np.clip(result, 0, 255).astype(np.float32)

    def fast_composite_active(self, bg_cache: np.ndarray) -> np.ndarray:
        """
        Sovrappone solo il layer attivo sul background pre-calcolato.
        Molto più veloce di composite() durante i tratti di disegno.
        """
        layer = self.active
        result = bg_cache.copy()   # float32
        canvas_h, canvas_w = result.shape[:2]
        lh, lw = layer.data.shape[:2]
        ox, oy = layer.offset_x, layer.offset_y
        cx1 = max(0, ox);           cy1 = max(0, oy)
        cx2 = min(canvas_w, ox + lw); cy2 = min(canvas_h, oy + lh)
        if cx2 > cx1 and cy2 > cy1:
            lx1 = cx1 - ox;  ly1 = cy1 - oy
            lx2 = lx1 + (cx2 - cx1); ly2 = ly1 + (cy2 - cy1)
            region = layer.data[ly1:ly2, lx1:lx2].astype(np.float32)
            if layer.mask is not None:
                mh, mw = layer.mask.shape
                mlx2 = min(mw, lx2); mly2 = min(mh, ly2)
                mask_r = layer.mask[ly1:mly2, lx1:mlx2].astype(np.float32) / 255.0
                region = region[:mask_r.shape[0], :mask_r.shape[1]].copy()
                region[:, :, 3] *= mask_r
                cx2 = cx1 + region.shape[1]
                cy2 = cy1 + region.shape[0]
            dst = result[cy1:cy2, cx1:cx2]
            result[cy1:cy2, cx1:cx2] = _composite_over(
                dst, region, layer.opacity, layer.blend_mode)
        return np.clip(result, 0, 255).astype(np.uint8)

    def _composite_nodes(self, nodes: list[NodeType], canvas_w: int, canvas_h: int,
                         override_path: Path | None = None,
                         override_data: np.ndarray | None = None,
                         skip_path: Path | None = None,
                         _depth_prefix: Path = ()) -> np.ndarray:
        result = np.zeros((canvas_h, canvas_w, 4), dtype=np.float32)
        for i, node in enumerate(nodes):
            current_path = (*_depth_prefix, i)
            if skip_path is not None and current_path == skip_path:
                continue   # escludi il layer attivo (per pre-composite background)
            if not node.visible:
                continue
            if isinstance(node, AdjustmentLayer):
                # Applica la regolazione al composito accumulato finora
                result = node.apply(result)
                continue
            if isinstance(node, LayerGroup):
                # Composta i figli in un buffer temporaneo
                if not node.children:
                    continue
                group_composite = self._composite_nodes(
                    node.children, canvas_w, canvas_h,
                    override_path=override_path, override_data=override_data,
                    skip_path=skip_path,
                    _depth_prefix=current_path,
                )
                group_f = group_composite.astype(np.float32)
                result = _composite_over(result, group_f, node.opacity, node.blend_mode)
            else:
                # Layer foglia
                if override_path is not None and current_path == override_path:
                    raw = override_data
                else:
                    raw = node.data
                lh, lw = raw.shape[:2]
                ox, oy = node.offset_x, node.offset_y
                cx1 = max(0, ox);          cy1 = max(0, oy)
                cx2 = min(canvas_w, ox + lw); cy2 = min(canvas_h, oy + lh)
                if cx2 <= cx1 or cy2 <= cy1:
                    continue
                lx1 = cx1 - ox;  ly1 = cy1 - oy
                lx2 = lx1 + (cx2 - cx1); ly2 = ly1 + (cy2 - cy1)
                region = raw[ly1:ly2, lx1:lx2].astype(np.float32)
                if node.mask is not None:
                    mh, mw = node.mask.shape
                    mlx2 = min(mw, lx2); mly2 = min(mh, ly2)
                    mask_r = node.mask[ly1:mly2, lx1:mlx2].astype(np.float32) / 255.0
                    region = region[:mask_r.shape[0], :mask_r.shape[1]].copy()
                    region[:, :, 3] *= mask_r
                    cx2 = cx1 + region.shape[1]
                    cy2 = cy1 + region.shape[0]
                dst = result[cy1:cy2, cx1:cx2]
                result[cy1:cy2, cx1:cx2] = _composite_over(
                    dst, region, node.opacity, node.blend_mode)
        return result

    # ── Snapshot ──────────────────────────────────────────────────────

    def snapshot(self) -> dict:
        return {
            "canvas_w": self._canvas_w,
            "canvas_h": self._canvas_h,
            "nodes": [n.snapshot() for n in self._nodes],
            "active_path": list(self._active_path),
        }

    @classmethod
    def from_snapshot(cls, snap: dict) -> "LayerStack":
        # Supporta snapshot vecchi (senza canvas_w/h) cercando il primo layer foglia
        if "canvas_w" in snap:
            stack = cls(canvas_w=snap["canvas_w"], canvas_h=snap["canvas_h"])
        else:
            # Fallback: leggi dal primo nodo (compatibilità con snapshot pre-fix)
            stack = cls()
            stack._nodes = [node_from_snapshot(d) for d in snap["nodes"]]
            stack._active_path = tuple(snap["active_path"])
            first_leaves = stack._all_leaf_paths()
            if first_leaves:
                first = stack.get_node(first_leaves[0])
                if isinstance(first, Layer):
                    stack._canvas_h, stack._canvas_w = first.data.shape[:2]
            return stack
        stack._nodes = [node_from_snapshot(d) for d in snap["nodes"]]
        stack._active_path = tuple(snap["active_path"])
        return stack

    # ── Helpers privati ───────────────────────────────────────────────

    def _count_all(self) -> int:
        return len(self._all_leaf_paths())

    def _fix_active_after_remove(self, removed_path: Path) -> None:
        leaves = self._all_leaf_paths()
        if not leaves:
            return
        # Se il path attivo era dentro il nodo rimosso o dopo di esso, aggiusta
        if (self._active_path == removed_path or
                (len(self._active_path) > len(removed_path) and
                 self._active_path[:len(removed_path)] == removed_path)):
            # Prendi il leaf precedente o il primo disponibile
            idx = removed_path[-1]
            parent_prefix = removed_path[:-1]
            # Cerca il leaf più vicino
            candidates = [p for p in leaves
                          if len(p) == 1 or p[:len(parent_prefix)] == parent_prefix]
            if candidates:
                self._active_path = candidates[0]
            else:
                self._active_path = leaves[0]
        # Shift degli indici al livello del nodo rimosso
        elif (len(self._active_path) >= len(removed_path) and
              self._active_path[:len(removed_path) - 1] == removed_path[:-1] and
              self._active_path[len(removed_path) - 1] > removed_path[-1]):
            # L'active era dopo il nodo rimosso nello stesso parent
            shifted = list(self._active_path)
            shifted[len(removed_path) - 1] -= 1
            self._active_path = tuple(shifted)


# ── Compositor ────────────────────────────────────────────────────────────────

def _blend_rgb(src: np.ndarray, dst: np.ndarray, mode: str) -> np.ndarray:
    s = src / 255.0; d = dst / 255.0
    if   mode == "multiply":   b = s * d
    elif mode == "screen":     b = 1 - (1 - s) * (1 - d)
    elif mode == "overlay":    b = np.where(d < 0.5, 2*s*d, 1-2*(1-s)*(1-d))
    elif mode == "soft_light": b = np.where(s < 0.5,
                                    d-(1-2*s)*d*(1-d), d+(2*s-1)*((2*d*d+2*d-1)-d))
    elif mode == "hard_light": b = np.where(s < 0.5, 2*s*d, 1-2*(1-s)*(1-d))
    elif mode == "difference": b = np.abs(s - d)
    elif mode == "add":        b = np.clip(s + d, 0, 1)
    elif mode == "subtract":   b = np.clip(d - s, 0, 1)
    elif mode == "darken":     b = np.minimum(s, d)
    elif mode == "lighten":    b = np.maximum(s, d)
    else:                      b = s
    return b * 255.0


def _composite_over(dst: np.ndarray, src: np.ndarray,
                    opacity: float, mode: str) -> np.ndarray:
    src_a = src[:, :, 3:4] * opacity / 255.0
    dst_a = dst[:, :, 3:4] / 255.0
    blended = _blend_rgb(src[:, :, :3], dst[:, :, :3], mode)
    out_a = src_a + dst_a * (1 - src_a)
    denom = np.maximum(out_a, 1e-8)
    out_rgb = (blended * src_a + dst[:, :, :3] * dst_a * (1 - src_a)) / denom
    result = np.zeros_like(dst)
    result[:, :, :3] = out_rgb
    result[:, :, 3:4] = out_a * 255.0
    return result
