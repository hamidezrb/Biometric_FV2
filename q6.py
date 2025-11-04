"""
ISO/IEC 29794-9 Quality Component 6 (Sharpness)
Finger Vascular Biometrics - Second Phalanx

Method (aligned with draft text used in your course):
- Compute edge strength using Sobel filters in four directions (0°, 45°, 90°, 135°).
- Combine responses (mean of absolute responses).
- Min–max normalize to 8-bit [0,255].
- Inside the foreground region R, count pixels with normalized sharpness > T (default T=100).
- Map the count to [0,100] using a scaling factor (default scale=0.006).
- Veto: if R is empty, Q6 = 0.

Returns:
    (Q6_score:int, strong_edge_count:int, threshold:int, scale:float)
"""

import cv2
import numpy as np
from typing import Tuple

def _sobel_4dir(gray: np.ndarray) -> np.ndarray:
    """Compute Sobel responses in 4 directions and return the average magnitude."""
    # Standard Sobel (0°/90°)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)   # 0°: vertical edges
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)   # 90°: horizontal edges

    # Diagonal Sobel-like kernels (45°, 135°)
    k45 = np.array([[ 0,  1,  2],
                    [-1,  0,  1],
                    [-2, -1,  0]], dtype=np.float32)
    k135 = np.array([[ 2,  1,  0],
                     [ 1,  0, -1],
                     [ 0, -1, -2]], dtype=np.float32)

    g45  = cv2.filter2D(gray, cv2.CV_32F, k45)
    g135 = cv2.filter2D(gray, cv2.CV_32F, k135)

    # Mean of absolute responses
    mag = (np.abs(gx) + np.abs(gy) + np.abs(g45) + np.abs(g135)) / 4.0
    return mag


def calculate_q6(R_mask: np.ndarray,
                 Grayscale_Image: np.ndarray,
                 threshold: int = 100,
                 scale: float = 0.006) -> Tuple[int, int, int, float]:
    """
    Q6 — Sharpness.

    Args:
        R_mask: binary mask (255=foreground, 0=background), same region as Q1.
        Grayscale_Image: original grayscale image (uint8 expected).
        threshold: normalized edge threshold in [0..255] to count "strong" edges (default 100).
        scale: scaling factor to map strong-edge count to [0..100] (default 0.006).

    Returns:
        (Q6_score, strong_edge_count, threshold, scale)
    """
    # Veto: invalid foreground
    fg = (R_mask == 255)
    if np.count_nonzero(fg) == 0:
        return 0, 0, threshold, scale

    # Ensure uint8 grayscale
    gray = Grayscale_Image
    if gray.dtype != np.uint8:
        gray = gray.astype(np.uint8)

    # 1) Sobel in four directions
    mag = _sobel_4dir(gray)

    # 2) Min–max normalize to [0,255] (float -> uint8)
    min_val = float(mag.min())
    max_val = float(mag.max())
    if max_val > min_val:
        mag_norm = ((mag - min_val) * (255.0 / (max_val - min_val)))
    else:
        mag_norm = np.zeros_like(mag, dtype=np.float32)
    mag_u8 = np.clip(mag_norm, 0, 255).astype(np.uint8)

    # 3) Count strong edges within foreground
    strong_edge_count = int(np.sum((mag_u8 > threshold) & fg))

    # 4) Map to [0,100] using scale (cap at 100)
    q6_raw = strong_edge_count * float(scale)
    Q6 = int(round(q6_raw))
    Q6 = max(0, min(100, Q6))

    return Q6, strong_edge_count, threshold, scale
