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


def extract_foreground_region(grayscale: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Extract foreground region R (Clause 5.2.1 Step a).

    The draft specifies extraction of the foreground region but does not define
    the segmentation algorithm on the Q1 page. This implementation uses:
    Otsu binarization, polarity resolution, and largest 8-connected component.

    Returns:
        R_mask: uint8 mask (255 = foreground)
        S_unoccluded: pixel count in R (unoccluded foreground area)
    """
    if grayscale is None or grayscale.size == 0:
        raise ValueError("grayscale image is empty")

    if grayscale.ndim != 2:
        raise ValueError("grayscale image must be 2-D")

    gray = grayscale if grayscale.dtype == np.uint8 else grayscale.astype(np.uint8)

    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask_inv = cv2.bitwise_not(mask)

    if int(np.sum(mask)) < int(np.sum(mask_inv)):
        mask = mask_inv

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    if num_labels <= 1:
        return np.zeros_like(gray, dtype=np.uint8), 0

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + int(np.argmax(areas))
    R_mask = np.where(labels == largest_label, 255, 0).astype(np.uint8)

    S_unoccluded = int(np.count_nonzero(R_mask == 255))
    return R_mask, S_unoccluded
