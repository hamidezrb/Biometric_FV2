"""
ISO/IEC 29794-9 Quality Component 8 (Total Vascular Length) — Clause 5.2.8.
"""

from __future__ import annotations
from typing import Optional, Tuple, Union
import numpy as np

from iso_constants import CaptureSite, DEFAULT_CAPTURE_SITE, get_capture_site_coefficients
from iso_foreground import is_foreground_region_valid
from vessel_utils import prepare_vessel_skeleton


def compute_q8_score(
    N_vessel: int,
    Lc: int,
) -> int:
    """
    Formula (15) — map N_vessel to [0, 100] and round.

    Q8 = MIN(100, ROUND(N_vessel / Lc * 100))
    """
    if Lc <= 0 or N_vessel < 0:
        return 0
    q8_raw = (float(N_vessel) / float(Lc)) * 100.0
    return max(0, min(100, int(round(q8_raw))))


def count_vessel_pixels_in_foreground(
    skel01: np.ndarray,
    foreground_mask: np.ndarray,
) -> int:
    """
    Step c: count pixels with value 1 in the thinned foreground region only.
    """
    if skel01.shape != foreground_mask.shape:
        raise ValueError("skeleton and foreground mask shape mismatch")
    in_fg = (skel01 == 1) & foreground_mask
    return int(np.count_nonzero(in_fg))


def calculate_q8(
    R_mask: np.ndarray,
    vein_img_or_path: Union[str, np.ndarray],
    capture_site: CaptureSite = DEFAULT_CAPTURE_SITE,
    Lc: Optional[int] = None,
    S_unoccluded: Optional[int] = None,
    *,
    iso_minimal: bool = True,
) -> Tuple[int, int, np.ndarray]:
    """
    ISO/IEC 29794-9 Clause 5.2.8 — total vascular length Q8.

    Step a): foreground from R_mask; invalid => Q8 = 0.
    Steps b–c): binarize within R, thin (algorithm in [5]), count N_vessel in R.
    Steps d–e): Table 2 Lc, Formula (15).
    """
    if R_mask is None:
        return 0, 0, np.zeros((1, 1), dtype=np.uint8)

    fg = (R_mask == 255)
    if S_unoccluded is None:
        S_unoccluded = int(np.count_nonzero(fg))

    if not is_foreground_region_valid(S_unoccluded):
        return 0, 0, np.zeros_like(R_mask, dtype=np.uint8)

    _, skel01 = prepare_vessel_skeleton(
        R_mask, vein_img_or_path, iso_minimal=iso_minimal
    )

    N_vessel = count_vessel_pixels_in_foreground(skel01, fg)
    effective_lc = (
        Lc if Lc is not None
        else get_capture_site_coefficients(capture_site)["LC"]
    )

    Q8 = compute_q8_score(N_vessel, effective_lc)
    return Q8, N_vessel, skel01
