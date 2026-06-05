"""
ISO/IEC 29794-9 Quality Component 5 (Information Entropy) — Clause 5.2.5.
"""

import numpy as np
from typing import Tuple

from iso_constants import EP_C_ENTROPY


def _shannon_entropy(values: np.ndarray, bit_depth: int = 8) -> float:
    """Shannon entropy H (bits) over grayscale levels present in R."""
    if values.size == 0:
        return 0.0
    L = 2 ** bit_depth
    hist = np.bincount(values.astype(np.uint8), minlength=L).astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 0.0
    p = hist / total
    nz = p > 0
    return float(-np.sum(p[nz] * np.log2(p[nz])))


def calculate_q5(
    R_mask: np.ndarray,
    Grayscale_Image: np.ndarray,
    bit_depth: int = 8,
    ep_c: float = EP_C_ENTROPY,
) -> Tuple[int, float]:
    """
    Formula (10): ep = H_bits / D, Q5 = min(100, round((ep / ep_c) * 100)).

    Invalid foreground => Q5 = 0.
    """
    fg = (R_mask == 255)
    if np.count_nonzero(fg) == 0:
        return 0, 0.0

    vals = Grayscale_Image[fg].astype(np.uint8)
    H_bits = _shannon_entropy(vals, bit_depth=bit_depth)

    if bit_depth <= 0 or ep_c <= 0:
        return 0, H_bits

    ep = H_bits / float(bit_depth)
    q5_raw = (ep / float(ep_c)) * 100.0
    Q5 = min(100, int(round(q5_raw)))
    Q5 = max(0, Q5)
    return Q5, H_bits
