import numpy as np
from typing import Tuple


def calculate_q4(
    R_mask: np.ndarray,
    Grayscale_Image: np.ndarray
) -> Tuple[int, float, float]:
    """
    Q4 — Equivalent Number of Looks (ENL)
    ISO/IEC 29794-9 Clause 5.2.4

    Returns:
        (Q4_score, sigma, g_mean)
    """

    fg = (R_mask == 255)
    N = int(np.count_nonzero(fg))
    if N == 0:
        return 0, 0.0, 0.0

    vals = Grayscale_Image[fg].astype(np.float64)

    g_mean = float(np.mean(vals))
    sigma = float(np.std(vals, ddof=0))

    if g_mean <= 0.0:
        return 0, sigma, g_mean

    ratio = sigma / g_mean
    q4_raw = (1.0 / (1.0 + ratio ** 2)) * 100.0
    Q4 = int(round(q4_raw))
    Q4 = max(0, min(100, Q4))

    return Q4, sigma, g_mean