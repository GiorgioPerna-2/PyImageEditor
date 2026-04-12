from __future__ import annotations
import numpy as np
from .layer import node_from_snapshot


class HistoryManager:
    def __init__(self, max_steps: int = 30) -> None:
        self._max_steps = max_steps
        self._stack: list[dict] = []
        self._composites: list[np.ndarray] = []   # thumbnail per lo storico visuale
        self._index: int = -1

    def push(self, snapshot: dict, composite: np.ndarray | None = None) -> None:
        if self._index < len(self._stack) - 1:
            self._stack = self._stack[:self._index + 1]
            self._composites = self._composites[:self._index + 1]
        self._stack.append(_deep_copy(snapshot))
        # Salva una copia ridotta del composite (max 256px lato lungo)
        if composite is not None:
            h, w = composite.shape[:2]
            self._composites.append(composite.copy())
        else:
            self._composites.append(None)
        self._index = len(self._stack) - 1
        if len(self._stack) > self._max_steps:
            overflow = len(self._stack) - self._max_steps
            self._stack = self._stack[overflow:]
            self._composites = self._composites[overflow:]
            self._index = len(self._stack) - 1

    def undo(self) -> dict | None:
        if not self.can_undo: return None
        self._index -= 1
        return _deep_copy(self._stack[self._index])

    def redo(self) -> dict | None:
        if not self.can_redo: return None
        self._index += 1
        return _deep_copy(self._stack[self._index])

    def jump_to(self, index: int) -> dict | None:
        """Salta direttamente a uno snapshot specifico (usato dallo storico visuale)."""
        if 0 <= index < len(self._stack):
            self._index = index
            return _deep_copy(self._stack[self._index])
        return None

    def clear(self) -> None:
        self._stack.clear()
        self._composites.clear()
        self._index = -1

    @property
    def can_undo(self) -> bool: return self._index > 0
    @property
    def can_redo(self) -> bool: return self._index < len(self._stack) - 1
    @property
    def current(self) -> dict | None:
        return _deep_copy(self._stack[self._index]) if self._index >= 0 else None
    @property
    def steps_count(self) -> int: return len(self._stack)
    @property
    def current_index(self) -> int: return self._index
    @property
    def composites(self) -> list:
        """Lista dei composites salvati (alcuni possono essere None)."""
        return self._composites


def _deep_copy(snap: dict) -> dict:
    """Copia profonda ricorsiva di uno snapshot LayerStack."""
    return {
        "nodes": [_deep_copy_node(n) for n in snap["nodes"]],
        "active_path": list(snap["active_path"]),
    }


def _deep_copy_node(d: dict) -> dict:
    if d.get("_type") == "group":
        return {
            "_type": "group",
            "name": d["name"],
            "children": [_deep_copy_node(c) for c in d["children"]],
            "visible": d["visible"],
            "opacity": d["opacity"],
            "blend_mode": d["blend_mode"],
        }
    # Layer foglia
    return {
        "_type": "layer",
        "data": d["data"].copy(),
        "name": d["name"],
        "visible": d["visible"],
        "opacity": d["opacity"],
        "blend_mode": d["blend_mode"],
        "offset_x": d.get("offset_x", 0),
        "offset_y": d.get("offset_y", 0),
        "mask": d["mask"].copy() if d.get("mask") is not None else None,
    }
