"""
ISO/IEC 29794-9 — foreground region R extraction (Clause 5.2.1 Step a).

Used by Q1 (effective area) and all metrics that consume R_mask.
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Tuple


def is_foreground_region_valid(S_unoccluded: int) -> bool:
    """
    ISO/IEC 29794-9 Clause 5.2.1 Step a) — foreground validity.

    The draft requires Q1 = 0 when the extracted foreground is not valid,
    e.g. contains zero pixels.

    TODO: The visible draft text also lists foreground that \"contains only noise\"
    as invalid; no operational noise criterion is defined on the Q1 page.
    """
    return S_unoccluded > 0


def _largest_connected_component(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    """Return the largest 8-connected component as a 255/0 mask and its area."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if num_labels <= 1:
        return np.zeros_like(mask, dtype=np.uint8), 0

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + int(np.argmax(areas))
    R_mask = np.where(labels == largest_label, 255, 0).astype(np.uint8)
    S_foreground = int(stats[largest_label, cv2.CC_STAT_AREA])
    return R_mask, S_foreground


def _select_subject_polarity(
    grayscale: np.ndarray,
    otsu_mask: np.ndarray,
) -> np.ndarray:
    """
    Choose Otsu polarity so R is the anatomical subject, not the background.

    Vascular captures typically place the hand/finger as the brighter region on
    a dark background. When polarity is resolved by area alone, low-quality
    images can mis-label the entire background as foreground.

    Select the polarity whose largest connected component has the higher mean
    intensity (the illuminated subject rather than the dark surround).
    """
    mask_inv = cv2.bitwise_not(otsu_mask)
    candidates = []

    for candidate_mask in (otsu_mask, mask_inv):
        R_candidate, area = _largest_connected_component(candidate_mask)
        if area <= 0:
            continue
        mean_intensity = float(np.mean(grayscale[R_candidate == 255]))
        candidates.append((mean_intensity, area, R_candidate))

    if not candidates:
        return np.zeros_like(otsu_mask, dtype=np.uint8)

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0][2]


def extract_foreground_region(grayscale: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Extract foreground region R (Clause 5.2.1 Step a).

    Region inside the finger/hand contour:
      1) Otsu binarization
      2) Polarity: select the subject (higher mean intensity), not background
      3) Largest 8-connected component

    Returns:
        R_mask: uint8 mask (255 = foreground)
        S_foreground: pixel count in R (total foreground area, before occlusion)
    """
    if grayscale is None or grayscale.size == 0:
        raise ValueError("grayscale image is empty")

    if grayscale.ndim != 2:
        raise ValueError("grayscale image must be 2-D")

    gray = grayscale if grayscale.dtype == np.uint8 else grayscale.astype(np.uint8)

    _, otsu_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    R_mask = _select_subject_polarity(gray, otsu_mask)

    S_foreground = int(np.count_nonzero(R_mask == 255))
    return R_mask, S_foreground
