"""
core/free_transform.py
----------------------
Trasformazione libera di un layer: scala, rotazione, skew.
Restituisce sempre un nuovo array RGBA uint8 con cv2.warpAffine/warpPerspective.
"""
from __future__ import annotations
import math
import numpy as np
import cv2


def apply_transform(
    data: np.ndarray,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    angle_deg: float = 0.0,
    skew_x: float = 0.0,    # in gradi: inclinazione orizzontale
    skew_y: float = 0.0,    # in gradi: inclinazione verticale
    pivot_x: float | None = None,   # punto di rotazione/scala in px (default centro)
    pivot_y: float | None = None,
    expand: bool = True,     # se True, allarga il canvas per non tagliare
) -> np.ndarray:
    """
    Applica una trasformazione affine al layer data (RGBA uint8).
    Restituisce un nuovo array (eventualmente di dimensioni diverse se expand=True).
    """
    h, w = data.shape[:2]
    cx = pivot_x if pivot_x is not None else w / 2.0
    cy = pivot_y if pivot_y is not None else h / 2.0

    # Costruisce la matrice affine combinando le trasformazioni
    # Ordine: sposta al pivot → scala → ruota → skew → torna all'origine
    rad = math.radians(angle_deg)
    cos_r = math.cos(rad);  sin_r = math.sin(rad)
    skx = math.tan(math.radians(skew_x))
    sky = math.tan(math.radians(skew_y))

    # Matrice 2x3 affine:
    #   M = T(pivot) · Rot · Scale · Skew · T(-pivot)
    # Costruiamo per composizione esplicita
    M = np.array([
        [scale_x * cos_r - sky * scale_y * sin_r,
         -scale_x * sin_r - sky * scale_y * cos_r + skx * scale_x * cos_r,
         0.0],
        [scale_x * sin_r + sky * scale_y * cos_r,
          scale_x * cos_r * sky + scale_y * cos_r - skx * scale_x * sin_r,  # nota: semplificato
         0.0],
    ], dtype=np.float64)

    # Matrice corretta senza cross-term
    M = np.float64([
        [cos_r * scale_x + skx * (-sin_r * scale_y),
         -sin_r * scale_x + skx * (cos_r * scale_y),
         0],
        [sin_r * scale_x + sky * cos_r * scale_y,
          cos_r * scale_x - sky * sin_r * scale_y,  # impreciso ma semplice
         0],
    ])

    # Approccio più pulito: usa getRotationMatrix2D + skew separata
    M_rot = cv2.getRotationMatrix2D((cx, cy), -angle_deg, 1.0)  # cv2 usa gradi CW
    M_scale = np.float64([[scale_x, 0, cx * (1 - scale_x)],
                           [0, scale_y, cy * (1 - scale_y)]])
    M_skew = np.float64([[1, math.tan(math.radians(skew_x)), 0],
                          [math.tan(math.radians(skew_y)), 1, 0]])

    # Combina: applica prima skew, poi scala, poi ruota
    # Usa matrici 3x3 omogenee per la composizione
    def to3x3(m):
        r = np.eye(3, dtype=np.float64)
        r[:2, :] = m
        return r

    M_combined = to3x3(M_rot) @ to3x3(M_scale) @ to3x3(M_skew)
    M_affine = M_combined[:2, :]   # torna a 2x3

    if expand:
        # Calcola le dimensioni del canvas risultante trasformando i 4 angoli
        corners = np.float64([[0, 0], [w, 0], [w, h], [0, h]])
        ones = np.ones((4, 1))
        corners_h = np.hstack([corners, ones])   # (4, 3)
        transformed = (M_affine @ corners_h.T).T  # (4, 2)
        min_x = transformed[:, 0].min()
        min_y = transformed[:, 1].min()
        max_x = transformed[:, 0].max()
        max_y = transformed[:, 1].max()
        new_w = int(math.ceil(max_x - min_x))
        new_h = int(math.ceil(max_y - min_y))
        # Traslazione per portare tutto nel quadrante positivo
        M_affine[0, 2] -= min_x
        M_affine[1, 2] -= min_y
    else:
        new_w, new_h = w, h

    result = cv2.warpAffine(
        data, M_affine, (max(1, new_w), max(1, new_h)),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    return result


def transform_matrix(
    w: int, h: int,
    scale_x: float, scale_y: float,
    angle_deg: float, skew_x: float, skew_y: float,
    pivot_x: float, pivot_y: float,
) -> np.ndarray:
    """
    Restituisce solo la matrice 2×3 senza applicarla — utile per l'anteprima canvas.
    """
    M_rot = cv2.getRotationMatrix2D((pivot_x, pivot_y), -angle_deg, 1.0)
    M_scale = np.float64([[scale_x, 0, pivot_x * (1 - scale_x)],
                           [0, scale_y, pivot_y * (1 - scale_y)]])
    M_skew = np.float64([[1, math.tan(math.radians(skew_x)), 0],
                          [math.tan(math.radians(skew_y)), 1, 0]])

    def to3x3(m):
        r = np.eye(3, dtype=np.float64)
        r[:2, :] = m
        return r

    M_combined = to3x3(M_rot) @ to3x3(M_scale) @ to3x3(M_skew)
    return M_combined[:2, :]
