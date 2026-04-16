"""
core/magic_wand.py
------------------
Strumento bacchetta magica: selezione per colore simile a partire da un punto.
Produce una maschera bool (H, W) come gli altri strumenti di selezione.

Algoritmo:
  1. Maschera di similarità vettorizzata (distanza euclidea RGBA, numpy).
  2. Se contiguous=True, si usa cv2.connectedComponents (C++) per estrarre
     solo la regione connessa al pixel scelto — molto più veloce del BFS
     Python puro che bloccava il thread UI.
  3. Anti-aliasing opzionale via GaussianBlur.
"""
from __future__ import annotations

import cv2
import numpy as np


# Soglia: distanza euclidea su 4 canali; il massimo teorico è sqrt(4)*255 ≈ 510.
# tolerance=32 → thr≈64  (comportamento identico alla versione precedente)
_SCALE = 2.0


def magic_wand_select(
    rgba: np.ndarray,
    x: int, y: int,
    tolerance: int = 32,
    contiguous: bool = True,
    anti_alias: bool = True,
    sample_merged: bool = False,   # riservato per implementazione futura
) -> np.ndarray:
    """
    Seleziona i pixel simili al pixel (x, y).

    Parameters
    ----------
    rgba        : array RGBA uint8 (H, W, 4)
    x, y        : coordinate del punto di campionamento (pixel immagine)
    tolerance   : soglia di distanza RGBA [0-255]
    contiguous  : se True seleziona solo i pixel contigui al punto di click;
                  se False seleziona tutti i pixel simili nell'intera immagine
    anti_alias  : sfuma leggermente il bordo della selezione

    Returns
    -------
    mask : bool array (H, W), True = pixel selezionato
    """
    h, w = rgba.shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        return np.zeros((h, w), dtype=bool)

    target = rgba[y, x].astype(np.int32)

    # ── 1. Maschera di similarità globale (numpy, vettorizzata) ──────
    similarity = _global_select(rgba, target, tolerance)

    # ── 2. Eventuale restrizione alla regione contigua ───────────────
    if contiguous:
        mask = _extract_connected(similarity, x, y)
    else:
        mask = similarity

    # ── 3. Anti-aliasing sul bordo ───────────────────────────────────
    if anti_alias:
        m = mask.astype(np.float32)
        m = cv2.GaussianBlur(m, (3, 3), 0.8)
        mask = m > 0.5

    return mask


# ── Funzioni di supporto ─────────────────────────────────────────────

def _global_select(rgba: np.ndarray,
                   target: np.ndarray,
                   tolerance: int) -> np.ndarray:
    """Maschera bool dei pixel entro 'tolerance' dal colore target (numpy)."""
    thr = float(tolerance) * _SCALE
    diff = rgba.astype(np.int32) - target[np.newaxis, np.newaxis, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    return dist <= thr


def _extract_connected(similarity: np.ndarray,
                        sx: int, sy: int) -> np.ndarray:
    """
    Dato il seed (sx, sy), restituisce solo i pixel della regione connessa
    che lo contiene, usando cv2.connectedComponents (C++, 4-connectivity).
    """
    if not similarity[sy, sx]:
        # Il pixel seme non soddisfa la soglia → selezione vuota
        return np.zeros_like(similarity)

    # cv2.connectedComponents vuole uint8 {0,255}
    src = similarity.astype(np.uint8) * 255
    # connectivity=4 coerente con il precedente BFS 4-direzionale
    _, labels = cv2.connectedComponents(src, connectivity=4)

    seed_label = labels[sy, sx]
    return labels == seed_label
