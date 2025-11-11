"""
ISO/IEC 29794-9 Quality Component 6 (Sharpness)
Clause 5.2.6 — official multi-Sobel implementation.

Steps:
  a) Extract R (veto if invalid)
  b) Convolve with 4 Sobel kernels (0°,45°,90°,135°)
  c) Average absolute responses → W_mean
  d) Min–max normalize to [0,255]
  e) Count N100 = pixels >100 inside R
  f) Q6 = min(100, round(g_c * N100)),  g_c = 0.006
"""

import numpy as np
import cv2
from typing import Tuple

def calculate_q6(R_mask: np.ndarray,
                 Grayscale_Image: np.ndarray,
                 g_c: float = 0.006,
                 threshold: int = 100) -> Tuple[int, int]:
    """Return (Q6_score, N100)."""
    fg = (R_mask == 255)
    if np.count_nonzero(fg) == 0:
        return 0, 0

    img = Grayscale_Image.astype(np.float32)
    img[~fg] = 0.0  # zero outside region

    # Four Sobel kernels (0°,45°,90°,135°)
    k0   = np.array([[ 1,  2,  1],
                     [ 0,  0,  0],
                     [-1, -2, -1]], np.float32)
    k45  = np.array([[ 2,  1,  0],
                     [ 1,  0, -1],
                     [ 0, -1, -2]], np.float32)
    k90  = np.array([[ 1,  0, -1],
                     [ 2,  0, -2],
                     [ 1,  0, -1]], np.float32)
    k135 = np.array([[ 0, -1, -2],
                     [ 1,  0, -1],
                     [ 2,  1,  0]], np.float32)

    I0   = cv2.filter2D(img, cv2.CV_32F, k0)
    I45  = cv2.filter2D(img, cv2.CV_32F, k45)
    I90  = cv2.filter2D(img, cv2.CV_32F, k90)
    I135 = cv2.filter2D(img, cv2.CV_32F, k135)

    # Average magnitude
    W_mean = (np.abs(I0) + np.abs(I45) + np.abs(I90) + np.abs(I135)) / 4.0

    # Normalize W_mean to [0,255]
    min_val, max_val = float(W_mean.min()), float(W_mean.max())
    if max_val > min_val:
        W_norm = (W_mean - min_val) * (255.0 / (max_val - min_val))
    else:
        W_norm = np.zeros_like(W_mean)
    W_norm = np.clip(W_norm, 0, 255).astype(np.uint8)

    # Count strong-edge pixels (>100) inside R
    N100 = int(np.sum((W_norm > threshold) & fg))

    # Compute Q6
    Q6 = int(round(g_c * N100))
    Q6 = max(0, min(100, Q6))

    return Q6, N100
