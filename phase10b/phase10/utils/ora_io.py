"""
utils/ora_io.py
---------------
Export/import del formato OpenRaster (.ora) — compatibile con Krita, GIMP,
MyPaint e altri editor che supportano lo standard.

Spec: https://www.openraster.org/

Struttura dello ZIP:
  mimetype                   (non compresso, prima entry)
  stack.xml                  (struttura layer in XML)
  data/layer_NNN.png         (dati RGBA di ogni layer)
  Thumbnails/thumbnail.png   (anteprima 256×256)
  mergedimage.png            (composite completo — richiesto dallo standard)

Note di compatibilità:
- I blend mode sono mappati dallo spazio nomi SVG usato da ORA ai nostri nomi interni.
- I LayerGroup vengono esportati come <stack> in ORA.
- Le maschere non fanno parte dello standard ORA base, vengono ignorate in export.
- In import, i layer fuori dai <stack> vengono letti come Layer piatti.
"""
from __future__ import annotations

import io
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path as FilePath

import numpy as np
from PIL import Image

from core.layer import Layer, LayerGroup, NodeType
from core.layer_stack import LayerStack

THUMBNAIL_MAX = 256

# Mappa blend mode interni → SVG compositing operator (ORA)
_TO_ORA: dict[str, str] = {
    "normal":     "svg:src-over",
    "multiply":   "svg:multiply",
    "screen":     "svg:screen",
    "overlay":    "svg:overlay",
    "soft_light": "svg:soft-light",
    "hard_light": "svg:hard-light",
    "difference": "svg:difference",
    "darken":     "svg:darken",
    "lighten":    "svg:lighten",
    # Questi non hanno equivalente ORA standard — fallback a normal
    "add":      "svg:src-over",
    "subtract": "svg:src-over",
}

# Mappa inversa ORA → interni
_FROM_ORA: dict[str, str] = {
    "svg:src-over":   "normal",
    "svg:multiply":   "multiply",
    "svg:screen":     "screen",
    "svg:overlay":    "overlay",
    "svg:soft-light": "soft_light",
    "svg:hard-light": "hard_light",
    "svg:difference": "difference",
    "svg:darken":     "darken",
    "svg:lighten":    "lighten",
    # Varianti senza prefisso (alcune app)
    "multiply":   "multiply",
    "screen":     "screen",
    "overlay":    "overlay",
    "soft-light": "soft_light",
    "hard-light": "hard_light",
    "difference": "difference",
    "darken":     "darken",
    "lighten":    "lighten",
    "normal":     "normal",
}


# ── Export ────────────────────────────────────────────────────────────────────

def save_ora(stack: LayerStack, path: str | FilePath) -> None:
    """Esporta il LayerStack come file .ora."""
    cw, ch = stack.canvas_size
    counter = [0]
    layer_pngs: list[tuple[str, np.ndarray]] = []   # (zip_path, data)

    def build_xml_nodes(nodes: list[NodeType], parent: ET.Element) -> None:
        # ORA: index 0 = bottom, ma in XML viene scritto dal basso verso l'alto
        # (il primo figlio nell'XML è il layer più in alto nello stack visivo)
        for node in reversed(nodes):
            if isinstance(node, LayerGroup):
                stack_el = ET.SubElement(parent, "stack")
                stack_el.set("name", node.name)
                stack_el.set("visibility", "visible" if node.visible else "hidden")
                stack_el.set("opacity", f"{node.opacity:.6f}")
                stack_el.set("composite-op", _TO_ORA.get(node.blend_mode, "svg:src-over"))
                build_xml_nodes(node.children, stack_el)
            else:
                idx = counter[0]; counter[0] += 1
                zip_path = f"data/layer_{idx:04d}.png"
                layer_pngs.append((zip_path, node.data))
                layer_el = ET.SubElement(parent, "layer")
                layer_el.set("name", node.name)
                layer_el.set("src", zip_path)
                layer_el.set("visibility", "visible" if node.visible else "hidden")
                layer_el.set("opacity", f"{node.opacity:.6f}")
                layer_el.set("composite-op", _TO_ORA.get(node.blend_mode, "svg:src-over"))
                layer_el.set("x", str(node.offset_x))
                layer_el.set("y", str(node.offset_y))

    # Costruisci XML
    root = ET.Element("image")
    root.set("version", "0.0.3")
    root.set("w", str(cw))
    root.set("h", str(ch))
    stack_root = ET.SubElement(root, "stack")
    build_xml_nodes(stack.nodes, stack_root)
    xml_bytes = ET.tostring(root, encoding="unicode", xml_declaration=False)
    xml_bytes = '<?xml version=\'1.0\' encoding=\'UTF-8\'?>\n' + xml_bytes

    # Composite + thumbnail
    composite = stack.composite()
    merged_buf = io.BytesIO()
    Image.fromarray(composite, "RGBA").save(merged_buf, format="PNG")

    thumb = Image.fromarray(composite, "RGBA")
    thumb.thumbnail((THUMBNAIL_MAX, THUMBNAIL_MAX), Image.LANCZOS)
    thumb_buf = io.BytesIO()
    thumb.save(thumb_buf, format="PNG")

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        # "mimetype" DEVE essere la prima entry e NON compressa (spec ORA)
        zf.writestr(zipfile.ZipInfo("mimetype"), "image/openraster")
        zf.writestr("stack.xml", xml_bytes)
        zf.writestr("mergedimage.png", merged_buf.getvalue())
        zf.writestr("Thumbnails/thumbnail.png", thumb_buf.getvalue())
        for zip_path, data in layer_pngs:
            buf = io.BytesIO()
            Image.fromarray(data, "RGBA").save(buf, format="PNG")
            zf.writestr(zip_path, buf.getvalue())


# ── Import ────────────────────────────────────────────────────────────────────

def load_ora(path: str | FilePath) -> LayerStack:
    """Carica un file .ora e restituisce un LayerStack."""
    with zipfile.ZipFile(path, "r") as zf:
        names = set(zf.namelist())

        # Verifica mimetype
        if "mimetype" in names:
            mt = zf.read("mimetype").decode("utf-8", errors="replace").strip()
            if mt != "image/openraster":
                raise ValueError(f"mimetype non valido: {mt!r}")

        xml_bytes = zf.read("stack.xml")
        root = ET.fromstring(xml_bytes)

        cw = int(root.get("w", 0))
        ch = int(root.get("h", 0))
        if cw <= 0 or ch <= 0:
            raise ValueError("Dimensioni canvas non valide in stack.xml")

        def read_png_rgba(src: str) -> np.ndarray:
            data = np.asarray(Image.open(io.BytesIO(zf.read(src))).convert("RGBA"),
                              dtype=np.uint8)
            return data

        def parse_nodes(xml_parent: ET.Element) -> list:
            nodes = []
            children = list(xml_parent)
            # ORA: primo figlio = layer più in alto; invertiamo per avere index 0 = bottom
            for el in reversed(children):
                tag = el.tag.lower()
                name = el.get("name", "Layer")
                visible = el.get("visibility", "visible") != "hidden"
                opacity = float(el.get("opacity", "1.0"))
                comp_op = el.get("composite-op", "svg:src-over")
                blend = _FROM_ORA.get(comp_op, "normal")

                if tag == "stack":
                    children_nodes = parse_nodes(el)
                    nodes.append(LayerGroup(
                        name=name, visible=visible,
                        opacity=opacity, blend_mode=blend,
                        children=children_nodes,
                    ))
                elif tag == "layer":
                    src = el.get("src", "")
                    if src not in names:
                        continue   # layer mancante — salta
                    data = read_png_rgba(src)
                    # Ridimensiona al canvas se necessario (alcuni file ORA hanno layer > canvas)
                    lh, lw = data.shape[:2]
                    ox = int(el.get("x", 0))
                    oy = int(el.get("y", 0))
                    nodes.append(Layer(
                        data=data, name=name,
                        visible=visible, opacity=opacity, blend_mode=blend,
                        offset_x=ox, offset_y=oy,
                    ))
            return nodes

        xml_stack = root.find("stack")
        if xml_stack is None:
            raise ValueError("Nessun <stack> radice trovato in stack.xml")

        nodes = parse_nodes(xml_stack)
        if not nodes:
            raise ValueError("Nessun layer trovato nel file ORA")

        stack = LayerStack(canvas_w=cw, canvas_h=ch)
        stack._nodes = nodes
        # Imposta active al primo layer foglia
        leaves = stack._all_leaf_paths()
        stack._active_path = leaves[0] if leaves else (0,)
        return stack
