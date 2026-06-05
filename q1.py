"""
ISO/IEC 29794-9 Quality Component 1 (Effective Area) — Clause 5.2.1.
"""

import cv2
import numpy as np
from typing import Optional, Tuple

from iso_constants import CaptureSite, DEFAULT_CAPTURE_SITE, get_capture_site_coefficients
from iso_foreground import extract_foreground_region, is_foreground_region_valid


def compute_q1_score(S_unoccluded: int, Sc: int) -> int:
    """
    ISO/IEC 29794-9 Formula (1):

        Q1 = MIN(100, ROUND(S_unoccluded / Sc * 100))
    """
    if not is_foreground_region_valid(S_unoccluded) or Sc <= 0:
        return 0
    return min(100, int(round(S_unoccluded / float(Sc) * 100.0)))


def calculate_q1(
    image_path: str,
    capture_site: CaptureSite = DEFAULT_CAPTURE_SITE,
    Sc: Optional[int] = None,
) -> Tuple[int, int, np.ndarray, np.ndarray]:
    """
    Effective area Q1 per Clause 5.2.1.

    Step b): Sc from Table 1 via capture_site (unless Sc overridden).
    Step c): Formula (1) via compute_q1_score.

    Returns (Q1_score, S_unoccluded, R_mask, Grayscale_Image).
    Invalid foreground => Q1 = 0.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        empty = np.zeros((1, 1), dtype=np.uint8)
        return 0, 0, empty, empty

    R_mask, S_unoccluded = extract_foreground_region(image)

    effective_sc = (
        Sc if Sc is not None
        else get_capture_site_coefficients(capture_site)["SC"]
    )

    Q1_score = compute_q1_score(S_unoccluded, effective_sc)
    return Q1_score, S_unoccluded, R_mask, image
