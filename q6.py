"""
ISO/IEC 29794-9 Quality Component 6 (Sharpness) — Clause 5.2.6.
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple

from iso_constants import GC_SHARPNESS
from iso_foreground import is_foreground_region_valid

# Figure 1 — Sobel operators (PWI draft 29794-9), exactly as in Clause 5.2.6.2.
ISO_SOBEL_KERNELS: Tuple[np.ndarray, ...] = (
    np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32),
    np.array([[2, 1, 0], [1, 0, -1], [0, -1, -2]], dtype=np.float32),
    np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32),
    np.array([[0, -1, -2], [1, 0, -1], [2, 1, 0]], dtype=np.float32),
)

N100_THRESHOLD = 100


def _convolve_iso_sobel_operators(gray_fg: np.ndarray) -> np.ndarray:
    """
    Steps c–d: convolve Figure 1 kernels (zero border), then Formula (11).

    W_mean = sum(|I_i|) / 4, i = 1..4
    """
    gray_f = gray_fg.astype(np.float32)
    convolved: List[np.ndarray] = []
    for kernel in ISO_SOBEL_KERNELS:
        I_i = cv2.filter2D(
            gray_f, cv2.CV_32F, kernel, borderType=cv2.BORDER_CONSTANT
        )
        convolved.append(I_i)
    return sum(np.abs(I_i) for I_i in convolved) / 4.0


def _minmax_normalize_to_uint8(response: np.ndarray) -> np.ndarray:
    """Step e: min-max normalize W_mean to [0, 255]."""
    mn = float(np.min(response))
    mx = float(np.max(response))
    if mx <= mn:
        return np.zeros(response.shape, dtype=np.uint8)
    scaled = (response - mn) / (mx - mn) * 255.0
    return scaled.astype(np.uint8)


def compute_q6_score(
    N100: int,
    S_unoccluded: int,
    gc: float = GC_SHARPNESS,
) -> int:
    """
    Formula (12) — map N100 to [0, 100] and round.

    Q6 = MIN(100, ROUND(N100 / (gc * S_unoccluded) * 100))

    The draft page shows the ratio form; the *100 scale factor matches
    Table-style quality mapping used elsewhere in ISO/IEC 29794-9 (cf. Q1).
    """
    if not is_foreground_region_valid(S_unoccluded) or gc <= 0:
        return 0
    q6_raw = (N100 / (gc * float(S_unoccluded))) * 100.0
    return max(0, min(100, int(round(q6_raw))))


def calculate_q6(
    R_mask: np.ndarray,
    Grayscale_Image: np.ndarray,
    S_unoccluded: Optional[int] = None,
    gc: float = GC_SHARPNESS,
    threshold: int = N100_THRESHOLD,
) -> Tuple[int, int]:
    """
    ISO/IEC 29794-9 Clause 5.2.6 — sharpness Q6.

    Step a): foreground from R_mask; invalid => Q6 = 0.
    Steps b–g): ISO Sobel kernels, Formula (11), min-max, N100, gc.
    Step h): Formula (12) via compute_q6_score.
    """
    fg = (R_mask == 255)
    if S_unoccluded is None:
        S_unoccluded = int(np.count_nonzero(fg))

    if not is_foreground_region_valid(S_unoccluded):
        return 0, 0

    gray = Grayscale_Image
    if gray.dtype != np.uint8:
        gray = gray.astype(np.uint8)

    # Step c: pixels outside foreground treated as zero before convolution.
    gray_fg = gray.copy()
    gray_fg[~fg] = 0

    w_mean = _convolve_iso_sobel_operators(gray_fg)
    w_norm = _minmax_normalize_to_uint8(w_mean)

    # Step f: pixels in W_mean with value > 100 (full image domain).
    N100 = int(np.count_nonzero(w_norm > threshold))

    Q6 = compute_q6_score(N100, S_unoccluded, gc=gc)
    return Q6, N100
