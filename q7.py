"""
ISO/IEC 29794-9 Quality Component 7 (Brightness Uniformity) — Clause 5.2.7.
"""

import numpy as np
from typing import Tuple, Optional

from iso_constants import Q7_BLOCK_NORMALIZER


def calculate_q7(
    R_mask: np.ndarray,
    Grayscale_Image: np.ndarray,
    g_mean: Optional[float] = None,
    block_size: int = 5,
) -> Tuple[int, float]:
    """
    Formula (14):
      Q7 = round((1 - (1/121) * sqrt((1/Nb) * sum((xi_mean - x_mean)^2))) * 100)

    5x5 blocks scanned top-to-bottom, left-to-right.
    Every block overlapping R is included (any foreground pixel).
    xi_mean uses only foreground pixels in the block.
    x_mean is the global mean over R.
    """
    fg = (R_mask == 255)
    if np.count_nonzero(fg) == 0:
        return 0, 0.0

    H, W = Grayscale_Image.shape
    bs = int(block_size)
    if bs <= 0 or bs > H or bs > W:
        return 0, 0.0

    x_mean = float(g_mean) if g_mean is not None else float(
        np.mean(Grayscale_Image[fg].astype(np.float64))
    )

    block_means = []
    for y in range(H - bs + 1):
        for x in range(W - bs + 1):
            block_fg = fg[y:y + bs, x:x + bs]
            if not np.any(block_fg):
                continue
            block_img = Grayscale_Image[y:y + bs, x:x + bs]
            xi_mean = float(np.mean(block_img[block_fg].astype(np.float64)))
            block_means.append(xi_mean)

    Nb = len(block_means)
    if Nb == 0:
        return 0, 0.0

    block_variance = float(np.mean((np.asarray(block_means) - x_mean) ** 2))
    sqrt_variance = float(np.sqrt(block_variance))
    q7_raw = (1.0 - (1.0 / float(Q7_BLOCK_NORMALIZER)) * sqrt_variance) * 100.0
    Q7_score = int(round(q7_raw))
    Q7_score = max(0, min(100, Q7_score))
    return Q7_score, block_variance
