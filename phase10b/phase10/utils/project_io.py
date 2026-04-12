"""
utils/project_io.py
-------------------
Salva e carica progetti Image Suite (.isp).

Formato: ZIP archive con
  project.json      — versione, dimensioni canvas, albero layer (senza pixel)
  layers/NNN.png    — dati RGBA di ogni Layer (PNG lossless, indicizzati in ordine
                      depth-first dell'albero)
  masks/NNN.png     — maschera grayscale (solo se presente)
  thumbnail.png     — anteprima 256×256 del composite

Il JSON descrive la struttura ricorsiva; ogni Layer ha un campo "_idx" che
corrisponde al numero nel nome file. I LayerGroup non hanno file associati.
"""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path as FilePath
from typing import Any

import numpy as np
from PIL import Image

from core.layer import Layer, LayerGroup, NodeType
from core.layer_stack import LayerStack

FORMAT_VERSION = 1
THUMBNAIL_MAX = 256


# ── Salvataggio ───────────────────────────────────────────────────────────────

def save_project(stack: LayerStack, path: str | FilePath) -> None:
    """Salva il progetto come file .isp (ZIP)."""
    counter = [0]   # contatore globale per i file layer
    layer_arrays: list[tuple[int, np.ndarray, np.ndarray | None]] = []
    # idx, data, mask

    def collect_nodes(nodes: list[NodeType]) -> list[dict]:
        result = []
        for node in nodes:
            if isinstance(node, Layer):
                idx = counter[0]; counter[0] += 1
                layer_arrays.append((idx, node.data, node.mask))
                result.append({
                    "_type": "layer",
                    "_idx": idx,
                    "name": node.name,
                    "visible": node.visible,
                    "opacity": node.opacity,
                    "blend_mode": node.blend_mode,
                    "offset_x": node.offset_x,
                    "offset_y": node.offset_y,
                    "has_mask": node.mask is not None,
                })
            else:
                result.append({
                    "_type": "group",
                    "name": node.name,
                    "visible": node.visible,
                    "opacity": node.opacity,
                    "blend_mode": node.blend_mode,
                    "children": collect_nodes(node.children),
                })
        return result

    tree = collect_nodes(stack.nodes)
    cw, ch = stack.canvas_size

    manifest = {
        "version": FORMAT_VERSION,
        "canvas_w": cw,
        "canvas_h": ch,
        "active_path": list(stack.active_path),
        "nodes": tree,
    }

    # Thumbnail
    composite = stack.composite()
    thumb_img = Image.fromarray(composite, "RGBA")
    thumb_img.thumbnail((THUMBNAIL_MAX, THUMBNAIL_MAX), Image.LANCZOS)
    thumb_buf = io.BytesIO()
    thumb_img.save(thumb_buf, format="PNG")

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        zf.writestr("project.json", json.dumps(manifest, indent=2))
        zf.writestr("thumbnail.png", thumb_buf.getvalue())

        for idx, data, mask in layer_arrays:
            # Layer data
            buf = io.BytesIO()
            Image.fromarray(data, "RGBA").save(buf, format="PNG")
            zf.writestr(f"layers/{idx:04d}.png", buf.getvalue())
            # Mask (se presente)
            if mask is not None:
                mbuf = io.BytesIO()
                Image.fromarray(mask, "L").save(mbuf, format="PNG")
                zf.writestr(f"masks/{idx:04d}.png", mbuf.getvalue())


# ── Caricamento ───────────────────────────────────────────────────────────────

def load_project(path: str | FilePath) -> LayerStack:
    """Carica un progetto .isp e restituisce il LayerStack."""
    with zipfile.ZipFile(path, "r") as zf:
        manifest: dict = json.loads(zf.read("project.json"))
        if manifest.get("version", 1) > FORMAT_VERSION:
            raise ValueError(
                f"Versione progetto {manifest['version']} non supportata "
                f"(massima: {FORMAT_VERSION})"
            )

        def read_png(zip_path: str) -> np.ndarray:
            return np.asarray(Image.open(io.BytesIO(zf.read(zip_path))), dtype=np.uint8)

        def build_nodes(dicts: list[dict]) -> list:
            nodes = []
            for d in dicts:
                if d["_type"] == "layer":
                    idx = d["_idx"]
                    data = read_png(f"layers/{idx:04d}.png")
                    # Assicura RGBA
                    if data.ndim == 2:
                        data = np.stack([data]*3 + [np.full_like(data, 255)], axis=-1)
                    elif data.shape[2] == 3:
                        alpha = np.full((*data.shape[:2], 1), 255, dtype=np.uint8)
                        data = np.concatenate([data, alpha], axis=-1)
                    mask = None
                    if d.get("has_mask"):
                        mask_path = f"masks/{idx:04d}.png"
                        if mask_path in zf.namelist():
                            mask = read_png(mask_path)
                            if mask.ndim == 3:
                                mask = mask[:, :, 0]
                    nodes.append(Layer(
                        data=data,
                        name=d["name"],
                        visible=d["visible"],
                        opacity=d["opacity"],
                        blend_mode=d["blend_mode"],
                        offset_x=d.get("offset_x", 0),
                        offset_y=d.get("offset_y", 0),
                        mask=mask,
                    ))
                else:
                    nodes.append(LayerGroup(
                        name=d["name"],
                        visible=d["visible"],
                        opacity=d["opacity"],
                        blend_mode=d["blend_mode"],
                        children=build_nodes(d["children"]),
                    ))
            return nodes

        cw = manifest["canvas_w"]
        ch = manifest["canvas_h"]
        stack = LayerStack(canvas_w=cw, canvas_h=ch)
        stack._nodes = build_nodes(manifest["nodes"])
        stack._active_path = tuple(manifest["active_path"])
        return stack
