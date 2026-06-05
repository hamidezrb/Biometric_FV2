"""
ISO/IEC 29794-9 — occlusion detection within foreground R (Clause 5.2.1).

Effective area uses the unoccluded foreground: pixels in R that are not
blocked by clothing, bandages, saturation, or other obstructions.
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Tuple


def max_representable_intensity(bit_depth: int = 8) -> int:
    """Maximum encoded intensity for the given bit depth."""
    return (1 << bit_depth) - 1


def detect_saturated_mask(
    grayscale: np.ndarray,
    foreground_mask: np.ndarray,
    bit_depth: int = 8,
) -> np.ndarray:
    """
    Pixels in R at the maximum representable intensity (overexposed / saturated).

    Returns uint8 mask (255 = saturated pixel within foreground).
    """
    max_val = max_representable_intensity(bit_depth)
    gray = grayscale if grayscale.dtype == np.uint8 else grayscale.astype(np.uint8)
    fg = foreground_mask == 255
    saturated = fg & (gray >= max_val)
    return np.where(saturated, 255, 0).astype(np.uint8)


def detect_occlusion_mask(
    grayscale: np.ndarray,
    foreground_mask: np.ndarray,
    bit_depth: int = 8,
) -> np.ndarray:
    """
    ISO 5.2.1 occlusion detector — union of obstruction classes within R.

    Currently implemented (deterministic, no dataset tuning):
      - Saturated / overexposed regions (intensity == max representable value)

    Clothing, bandage, and other non-saturated obstructions require the ISO
    reference implementation when published; they are not defined on the Q1 page.
    """
    return detect_saturated_mask(grayscale, foreground_mask, bit_depth=bit_depth)


def compute_unoccluded_foreground(
    foreground_mask: np.ndarray,
    occlusion_mask: np.ndarray,
) -> Tuple[np.ndarray, int, int, int]:
    """
    Derive unoccluded foreground from R and occlusion mask.

    Returns:
        R_unoccluded: uint8 mask (255 = valid effective-area pixel)
        S_foreground: pixel count in R
        S_occluded: occluded pixel count within R
        S_unoccluded: effective pixel count for Formula (1)
    """
    fg = foreground_mask == 255
    occ = (occlusion_mask == 255) & fg
    unocc = fg & ~occ

    S_foreground = int(np.count_nonzero(fg))
    S_occluded = int(np.count_nonzero(occ))
    S_unoccluded = int(np.count_nonzero(unocc))

    R_unoccluded = np.where(unocc, 255, 0).astype(np.uint8)
    return R_unoccluded, S_foreground, S_occluded, S_unoccluded
