"""
ISO/IEC 29794-9 Quality Component 1 (Effective Area) — Clause 5.2.1.
"""

import cv2
import numpy as np
from typing import NamedTuple, Optional, Tuple

from iso_constants import CaptureSite, DEFAULT_CAPTURE_SITE, get_capture_site_coefficients
from iso_foreground import extract_foreground_region, is_foreground_region_valid
from iso_occlusion import compute_unoccluded_foreground, detect_occlusion_mask


class Q1Result(NamedTuple):
    """Detailed Q1 computation outputs for audit and pipeline use."""

    Q1_score: int
    S_unoccluded: int
    R_foreground: np.ndarray
    grayscale: np.ndarray
    R_unoccluded: np.ndarray
    occlusion_mask: np.ndarray
    S_foreground: int
    S_occluded: int
    Sc: int
    q1_raw: float


def compute_q1_score(S_unoccluded: int, Sc: int) -> int:
    """
    ISO/IEC 29794-9 Formula (1):

        Q1 = MIN(100, ROUND(S_unoccluded / Sc * 100))
    """
    if not is_foreground_region_valid(S_unoccluded) or Sc <= 0:
        return 0
    return min(100, int(round(S_unoccluded / float(Sc) * 100.0)))


def compute_q1_raw(S_unoccluded: int, Sc: int) -> float:
    """Formula (1) value before rounding and capping."""
    if Sc <= 0:
        return 0.0
    return (S_unoccluded / float(Sc)) * 100.0


def calculate_q1_detailed(
    image_path: str,
    capture_site: CaptureSite = DEFAULT_CAPTURE_SITE,
    Sc: Optional[int] = None,
    bit_depth: int = 8,
) -> Q1Result:
    """
    Effective area Q1 per Clause 5.2.1 with full audit outputs.

    Step a): extract foreground R (finger/hand contour interior)
    Step a continued): detect occlusions within R; S_unoccluded = |R| - |occluded|
    Step b): Sc from Table 1 via capture_site (unless Sc overridden)
    Step c): Formula (1)
    """
    empty = np.zeros((1, 1), dtype=np.uint8)
    effective_sc = (
        Sc if Sc is not None
        else get_capture_site_coefficients(capture_site)["SC"]
    )

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return Q1Result(0, 0, empty, empty, empty, empty, 0, 0, effective_sc, 0.0)

    R_foreground, S_foreground = extract_foreground_region(image)
    occlusion_mask = detect_occlusion_mask(image, R_foreground, bit_depth=bit_depth)
    R_unoccluded, _, S_occluded, S_unoccluded = compute_unoccluded_foreground(
        R_foreground, occlusion_mask
    )

    q1_raw = compute_q1_raw(S_unoccluded, effective_sc)
    Q1_score = compute_q1_score(S_unoccluded, effective_sc)

    return Q1Result(
        Q1_score=Q1_score,
        S_unoccluded=S_unoccluded,
        R_foreground=R_foreground,
        grayscale=image,
        R_unoccluded=R_unoccluded,
        occlusion_mask=occlusion_mask,
        S_foreground=S_foreground,
        S_occluded=S_occluded,
        Sc=effective_sc,
        q1_raw=q1_raw,
    )


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
    R_mask is the foreground region R; S_unoccluded excludes occlusions.
    Invalid foreground => Q1 = 0.
    """
    result = calculate_q1_detailed(image_path, capture_site=capture_site, Sc=Sc)
    return result.Q1_score, result.S_unoccluded, result.R_foreground, result.grayscale
