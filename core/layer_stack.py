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

    def fast_composite_patch(
        self,
        bg_cache:   np.ndarray,
        canvas_img: np.ndarray,
        dirty_cx0:  int,
        dirty_cy0:  int,
        dirty_cx1:  int,
        dirty_cy1:  int,
    ) -> tuple[int, int, int, int]:
        """
        Aggiorna SOLO il patch sporco (dirty rect in coordinate canvas) di
        canvas_img sovrapponendo il layer attivo su bg_cache.

        Non copia mai l'intero bg_cache — opera solo sulla regione dirty,
        riducendo il lavoro da O(W×H) a O(brush_size²).

        Parameters
        ----------
        bg_cache    : float32 (canvas_H, canvas_W, 4) — sfondo pre-calcolato
        canvas_img  : uint8  (canvas_H, canvas_W, 4) — output aggiornato in-place
        dirty_cx0/1, dirty_cy0/1 : dirty rect già clippata ai bordi del canvas

        Returns
        -------
        (dirty_cx0, dirty_cy0, dirty_cx1, dirty_cy1)
        """
        layer = self.active
        canvas_h, canvas_w = bg_cache.shape[:2]
        ox, oy = layer.offset_x, layer.offset_y
        lh, lw = layer.data.shape[:2]

        # Intersezione dirty rect ∩ area del layer nel canvas
        lcx0 = max(0, ox);        lcy0 = max(0, oy)
        lcx1 = min(canvas_w, ox + lw);  lcy1 = min(canvas_h, oy + lh)

        ix0 = max(dirty_cx0, lcx0);  iy0 = max(dirty_cy0, lcy0)
        ix1 = min(dirty_cx1, lcx1);  iy1 = min(dirty_cy1, lcy1)

        if ix0 >= ix1 or iy0 >= iy1:
            # La dirty rect non si sovrappone al layer: copia solo il bg
            _blit_bg(bg_cache, canvas_img, dirty_cy0, dirty_cy1, dirty_cx0, dirty_cx1)
            return (dirty_cx0, dirty_cy0, dirty_cx1, dirty_cy1)

        # Coordinate layer locali
        llx0 = ix0 - ox;  lly0 = iy0 - oy
        llx1 = ix1 - ox;  lly1 = iy1 - oy

        bg_patch    = bg_cache[iy0:iy1, ix0:ix1]          # float32 view — no copy
        layer_patch = layer.data[lly0:lly1, llx0:llx1].astype(np.float32)

        if layer.mask is not None:
            mh, mw = layer.mask.shape
            mx0 = min(llx0, mw); my0 = min(lly0, mh)
            mx1 = min(llx1, mw); my1 = min(lly1, mh)
            if mx1 > mx0 and my1 > my0:
                mask_r = layer.mask[my0:my1, mx0:mx1].astype(np.float32) / 255.0
                ph = min(layer_patch.shape[0], mask_r.shape[0])
                pw = min(layer_patch.shape[1], mask_r.shape[1])
                layer_patch = layer_patch[:ph, :pw].copy()
                layer_patch[:, :, 3] *= mask_r[:ph, :pw]
                ix1 = ix0 + pw;  iy1 = iy0 + ph
                bg_patch = bg_cache[iy0:iy1, ix0:ix1]

        composited = _composite_over(bg_patch, layer_patch, layer.opacity, layer.blend_mode)
        canvas_img[iy0:iy1, ix0:ix1] = np.clip(composited, 0, 255).astype(np.uint8)

        # Aree della dirty rect fuori dal layer → copia solo bg
        for ry0, ry1, rx0, rx1 in _dirty_complement(
                dirty_cy0, dirty_cy1, dirty_cx0, dirty_cx1,
                iy0, iy1, ix0, ix1):
            _blit_bg(bg_cache, canvas_img, ry0, ry1, rx0, rx1)

        return (dirty_cx0, dirty_cy0, dirty_cx1, dirty_cy1)

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


    def composite_rect(
        self,
        cx0: int, cy0: int,
        cx1: int, cy1: int,
    ) -> np.ndarray:
        """
        Composita correttamente TUTTI i layer (ordine corretto, adjustment layers inclusi)
        per la sola regione [cx0:cx1, cy0:cy1] del canvas.

        O(n_layers × patch_area) invece di O(n_layers × W×H):
        per un pennello da 40px su un canvas 3000px, è ~5000× più veloce del full composite.

        Returns
        -------
        uint8 RGBA (cy1-cy0, cx1-cx0, 4)
        """
        ph = cy1 - cy0
        pw = cx1 - cx0
        if ph <= 0 or pw <= 0:
            return np.zeros((1, 1, 4), dtype=np.uint8)

        result = np.zeros((ph, pw, 4), dtype=np.float32)
        self._composite_nodes_rect(
            self._nodes, cx0, cy0, cx1, cy1, result, depth_prefix=())
        return np.clip(result, 0, 255).astype(np.uint8)

    def _composite_nodes_rect(
        self,
        nodes: list,
        cx0: int, cy0: int,
        cx1: int, cy1: int,
        result: np.ndarray,
        depth_prefix: tuple,
    ) -> None:
        """
        Itera i nodi e composta solo la regione [cx0:cx1, cy0:cy1] su 'result'.
        'result' è float32 (ph, pw, 4) con ph=cy1-cy0, pw=cx1-cx0.
        """
        ph, pw = result.shape[:2]
        for i, node in enumerate(nodes):
            if not node.visible:
                continue
            current_path = (*depth_prefix, i)

            if isinstance(node, AdjustmentLayer):
                # Gli adjustment layer operano su tutto il result accumulato finora.
                # Estrae il patch, applica il filtro, reinserisce.
                # Crea un canvas temporaneo di dimensione canonica, applica, ritaglia.
                tmp = np.zeros((self._canvas_h, self._canvas_w, 4), dtype=np.float32)
                tmp[cy0:cy1, cx0:cx1] = result
                tmp = node.apply(tmp)
                result[:] = tmp[cy0:cy1, cx0:cx1]
                continue

            if isinstance(node, LayerGroup):
                if not node.children:
                    continue
                group_buf = np.zeros((ph, pw, 4), dtype=np.float32)
                self._composite_nodes_rect(
                    node.children, cx0, cy0, cx1, cy1, group_buf, current_path)
                dst = result
                result[:] = _composite_over(dst, group_buf, node.opacity, node.blend_mode)
                continue

            # Layer foglia
            raw = node.data
            lh, lw = raw.shape[:2]
            ox, oy = node.offset_x, node.offset_y

            # Intersezione del patch richiesto con l'estensione del layer nel canvas
            # (tutto in coordinate canvas assolute)
            ix0 = max(cx0, max(0, ox))
            iy0 = max(cy0, max(0, oy))
            ix1 = min(cx1, min(self._canvas_w, ox + lw))
            iy1 = min(cy1, min(self._canvas_h, oy + lh))
            if ix0 >= ix1 or iy0 >= iy1:
                continue

            # Coordinate locali nel layer
            llx0 = ix0 - ox;  lly0 = iy0 - oy
            llx1 = ix1 - ox;  lly1 = iy1 - oy

            # Coordinate nel buffer result (che copre [cx0:cx1, cy0:cy1])
            rx0 = ix0 - cx0;  ry0 = iy0 - cy0
            rx1 = ix1 - cx0;  ry1 = iy1 - cy0

            region = raw[lly0:lly1, llx0:llx1].astype(np.float32)

            if node.mask is not None:
                mh, mw = node.mask.shape
                mlx1 = min(mw, llx1);  mly1 = min(mh, lly1)
                if mlx1 > llx0 and mly1 > lly0:
                    mask_r = node.mask[lly0:mly1, llx0:mlx1].astype(np.float32) / 255.0
                    ph_m = min(region.shape[0], mask_r.shape[0])
                    pw_m = min(region.shape[1], mask_r.shape[1])
                    region = region[:ph_m, :pw_m].copy()
                    region[:, :, 3] *= mask_r[:ph_m, :pw_m]
                    rx1 = rx0 + region.shape[1]
                    ry1 = ry0 + region.shape[0]

            dst_slice = result[ry0:ry1, rx0:rx1]
            result[ry0:ry1, rx0:rx1] = _composite_over(
                dst_slice, region, node.opacity, node.blend_mode)



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

def _dirty_complement(
    dy0: int, dy1: int, dx0: int, dx1: int,   # outer dirty rect
    iy0: int, iy1: int, ix0: int, ix1: int,   # inner (layer) rect
) -> list[tuple[int, int, int, int]]:
    """
    Restituisce le sotto-regioni della dirty rect che NON si sovrappongono
    con la regione del layer, utili per copiarvi il solo background.
    """
    rects: list[tuple[int, int, int, int]] = []
    if dy0 < iy0:
        rects.append((dy0, min(iy0, dy1), dx0, dx1))
    if iy1 < dy1:
        rects.append((max(iy1, dy0), dy1, dx0, dx1))
    mid_y0 = max(dy0, iy0); mid_y1 = min(dy1, iy1)
    if mid_y0 < mid_y1:
        if dx0 < ix0:
            rects.append((mid_y0, mid_y1, dx0, min(ix0, dx1)))
        if ix1 < dx1:
            rects.append((mid_y0, mid_y1, max(ix1, dx0), dx1))
    return rects


def _blit_bg(
    bg: np.ndarray, dst: np.ndarray,
    y0: int, y1: int, x0: int, x1: int,
) -> None:
    """Copia bg[y0:y1, x0:x1] → dst (conversione float32→uint8 inclusa)."""
    if y0 >= y1 or x0 >= x1:
        return
    dst[y0:y1, x0:x1] = np.clip(bg[y0:y1, x0:x1], 0, 255).astype(np.uint8)


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
