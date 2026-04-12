from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Union


BLEND_MODES: dict[str, str] = {
    "normal":     "Normale",
    "multiply":   "Moltiplica",
    "screen":     "Schermo",
    "overlay":    "Sovrapponi",
    "soft_light": "Luce soffusa",
    "hard_light": "Luce intensa",
    "difference": "Differenza",
    "add":        "Aggiungi",
    "subtract":   "Sottrai",
    "darken":     "Scurisci",
    "lighten":    "Schiarisci",
}


@dataclass
class Layer:
    data: np.ndarray          # RGBA uint8 (H, W, 4)
    name: str
    visible: bool = True
    opacity: float = 1.0
    blend_mode: str = "normal"
    offset_x: int = 0
    offset_y: int = 0
    mask: Optional[np.ndarray] = None   # grayscale uint8 (H, W)

    def clone(self) -> "Layer":
        return Layer(
            data=self.data.copy(), name=self.name,
            visible=self.visible, opacity=self.opacity,
            blend_mode=self.blend_mode,
            offset_x=self.offset_x, offset_y=self.offset_y,
            mask=self.mask.copy() if self.mask is not None else None,
        )

    def snapshot(self) -> dict:
        return {
            "_type": "layer",
            "data": self.data.copy(), "name": self.name,
            "visible": self.visible, "opacity": self.opacity,
            "blend_mode": self.blend_mode,
            "offset_x": self.offset_x, "offset_y": self.offset_y,
            "mask": self.mask.copy() if self.mask is not None else None,
        }

    @classmethod
    def from_snapshot(cls, d: dict) -> "Layer":
        return cls(
            data=d["data"].copy(), name=d["name"],
            visible=d["visible"], opacity=d["opacity"],
            blend_mode=d["blend_mode"],
            offset_x=d.get("offset_x", 0), offset_y=d.get("offset_y", 0),
            mask=d["mask"].copy() if d.get("mask") is not None else None,
        )

    # ── Mask helpers ──────────────────────────────────────────────────
    def add_mask(self, value: int = 255) -> None:
        h, w = self.data.shape[:2]
        self.mask = np.full((h, w), value, dtype=np.uint8)

    def remove_mask(self) -> None:
        self.mask = None

    def invert_mask(self) -> None:
        if self.mask is not None:
            self.mask = 255 - self.mask

    def apply_mask_permanently(self) -> None:
        if self.mask is None:
            return
        self.data = self.data.copy()
        self.data[:, :, 3] = (
            self.data[:, :, 3].astype(np.float32)
            * self.mask.astype(np.float32) / 255.0
        ).astype(np.uint8)
        self.mask = None


@dataclass
class LayerGroup:
    """
    Nodo intermedio dell'albero layer.
    I figli vengono compositati tra loro, poi il risultato
    viene blended nel parent con opacity e blend_mode del gruppo.
    """
    name: str
    children: list["NodeType"] = field(default_factory=list)
    visible: bool = True
    opacity: float = 1.0
    blend_mode: str = "normal"

    def clone(self) -> "LayerGroup":
        return LayerGroup(
            name=self.name,
            children=[c.clone() for c in self.children],
            visible=self.visible,
            opacity=self.opacity,
            blend_mode=self.blend_mode,
        )

    def snapshot(self) -> dict:
        return {
            "_type": "group",
            "name": self.name,
            "children": [c.snapshot() for c in self.children],
            "visible": self.visible,
            "opacity": self.opacity,
            "blend_mode": self.blend_mode,
        }

    @classmethod
    def from_snapshot(cls, d: dict) -> "LayerGroup":
        return cls(
            name=d["name"],
            children=[node_from_snapshot(c) for c in d["children"]],
            visible=d["visible"],
            opacity=d["opacity"],
            blend_mode=d["blend_mode"],
        )


# Union type per type hints
NodeType = Union[Layer, LayerGroup]


def node_from_snapshot(d: dict) -> NodeType:
    if d.get("_type") == "group":
        return LayerGroup.from_snapshot(d)
    return Layer.from_snapshot(d)
