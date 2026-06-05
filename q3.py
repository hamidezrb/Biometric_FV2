import numpy as np
from typing import Tuple


def calculate_q3(
    R_mask: np.ndarray,
    Grayscale_Image: np.ndarray,
    bit_depth: int = 8
) -> Tuple[int, float, float]:
    """
    Q3 — Contrast
    ISO/IEC 29794-9 Clause 5.2.3

    Returns:
        (Q3_score, sigma, g_mean)
    """

    fg = (R_mask == 255)
    N = int(np.count_nonzero(fg))
    if N == 0:
        return 0, 0.0, 0.0

    vals = Grayscale_Image[fg].astype(np.float64)

    g_mean = float(np.mean(vals))
    sigma = float(np.std(vals, ddof=0))

    normalization_factor = (2 ** bit_depth) / 4.0
    q3_raw = (sigma / normalization_factor) * 100.0
    Q3 = int(round(q3_raw))
    Q3 = max(0, min(100, Q3))

    return Q3, sigma, g_mean