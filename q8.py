"""
ISO/IEC 29794-9 (PWI draft) — Quality Component 8: Total vascular length
Clause 5.2.8, Formula (15)

Process:
1) Veto: if foreground region R is invalid => Q8 = 0
2) Binarize + thin the foreground region (vessels = 1, others = 0; thickness = 1 px)
3) Count vessel pixels Nv (value == 1) in the thinned image
4) Map to [0,100] using coefficient Lc (Table 2):
      Q8 = min(100, round((Nv / Lc) * 100))
"""

from __future__ import annotations
from typing import Tuple, Union
import cv2
import numpy as np
from vessel_utils import prepare_vessel_skeleton


def calculate_q8(
    R_mask: np.ndarray,
    vein_img_or_path: Union[str, np.ndarray],
    Lc: int = 600,
    *,
    min_area: int = 120,
    prune_iters: int = 30,
    min_skel_len: int = 120,
    keep_largest: bool = True
):
    if R_mask is None:
        return 0, 0, np.zeros((1, 1), dtype=np.uint8)

    fg = (R_mask == 255)
    if np.count_nonzero(fg) == 0:
        return 0, 0, np.zeros_like(R_mask, dtype=np.uint8)
    
    _, skel01 = prepare_vessel_skeleton(
        R_mask,
        vein_img_or_path,
        min_area=min_area,
        prune_iters=prune_iters,
        min_skel_len=min_skel_len,
        keep_largest=keep_largest,
    )

    N_vessel = int(np.count_nonzero(skel01))
    Q8 = min(100, int(round((N_vessel / Lc) * 100))) if Lc > 0 else 0
    return Q8, N_vessel, skel01
