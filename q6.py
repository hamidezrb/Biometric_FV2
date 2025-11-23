"""
ISO/IEC 29794-9 Quality Component 6 (Sharpness)
Finger Vascular Biometrics - Second Phalanx

This module re-exports the canonical Q6 implementation located in
quality_metrics.py so that existing imports continue to work.
"""

from typing import Tuple

from quality_metrics import calculate_q6 as _calculate_q6


def calculate_q6(R_mask, Grayscale_Image, S_unoccluded, gc: float = 0.006, threshold: int = 100) -> Tuple[int, int]:
    """
    Wrapper for backwards compatibility.
    """
    return _calculate_q6(R_mask, Grayscale_Image, S_unoccluded, gc=gc, threshold=threshold)
