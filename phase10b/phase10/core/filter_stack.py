from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .image_filters import (
    brightness, contrast, saturation, sharpen, blur,
    hue_shift, vibrance,
    apply_color_style,
    vignette, film_grain, soft_glow, clarity, dehaze,
)


@dataclass
class FilterSettings:
    # ── Tonali ──────────────────────────────
    brightness: int = 0         # -100 / +100
    contrast: int = 0           # -100 / +100
    saturation: int = 0         # -100 / +100
    sharpen: int = 0            # 0 / +100
    blur: int = 0               # 0 / +20
    hue_shift: int = 0          # -180 / +180
    vibrance: int = 0           # -100 / +100
    # ── Stile colore ────────────────────────
    color_style: str = "none"
    color_style_intensity: int = 100   # 0 / 100
    # ── Effetti fotografici ─────────────────
    vignette: int = 0           # 0 / 100
    grain: int = 0              # 0 / 100
    glow: int = 0               # 0 / 100
    clarity: int = 0            # -100 / +100
    dehaze: int = 0             # 0 / 100


class FilterStack:
    """
    Applica una catena di filtri non-distruttivi a un'immagine base.

    Pipeline:
        1. Tonali (brightness → contrast → saturation → hue_shift → vibrance)
        2. Dettaglio (sharpen → blur)
        3. Stile colore (preset)
        4. Effetti fotografici (clarity → dehaze → vignette → grain → glow)
    """

    def __init__(self) -> None:
        self.settings = FilterSettings()

    def update(self, **kwargs) -> None:
        for k, v in kwargs.items():
            if hasattr(self.settings, k):
                setattr(self.settings, k, v)

    def render(self, img: np.ndarray) -> np.ndarray:
        s = self.settings
        out = img.copy()

        # 1 — Tonali
        if s.brightness != 0:
            out = brightness(out, s.brightness)
        if s.contrast != 0:
            out = contrast(out, s.contrast)
        if s.saturation != 0:
            out = saturation(out, s.saturation)
        if s.hue_shift != 0:
            out = hue_shift(out, s.hue_shift)
        if s.vibrance != 0:
            out = vibrance(out, s.vibrance)

        # 2 — Dettaglio
        if s.sharpen != 0:
            out = sharpen(out, s.sharpen / 20.0)
        if s.blur != 0:
            out = blur(out, s.blur)

        # 3 — Stile colore
        if s.color_style != "none":
            out = apply_color_style(out, s.color_style, s.color_style_intensity)

        # 4 — Effetti fotografici
        if s.clarity != 0:
            out = clarity(out, s.clarity)
        if s.dehaze != 0:
            out = dehaze(out, s.dehaze)
        if s.vignette != 0:
            out = vignette(out, s.vignette)
        if s.grain != 0:
            out = film_grain(out, s.grain)
        if s.glow != 0:
            out = soft_glow(out, s.glow)

        return out
