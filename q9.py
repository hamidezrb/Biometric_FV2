"""
ISO/IEC 29794-9 (PWI draft) — Quality Component 9: Number of feature points
Clause 5.2.9, Formula (16)

Process:
1) Veto: if foreground region R is invalid => Q9 = 0
2) Binarize + thin (same as Q8) to a 1-pixel skeleton
3) For each skeleton pixel p=1, examine 8-neighborhood clockwise.
   Let T be the number of 0↔1 transitions in the circular sequence.
   - If T == 2 => endpoint
   - If T == 6 or 8 => intersection
4) N_fp = N_end + N_int
5) Map to [0,100] using coefficient Fc (Table 3):
      Q9 = min(100, round((N_fp / Fc) * 100))
"""

from __future__ import annotations
from typing import Tuple, Union
import numpy as np
import cv2
from vessel_utils import prepare_vessel_skeleton


def _transition_count_8nbr(img01: np.ndarray, y: int, x: int) -> int:
    """
    ISO: count 0↔1 transitions (total changes) in clockwise 8-neighborhood sequence.
    Endpoint => 2 transitions, intersection => 6 or 8 transitions.
    """
    p2 = img01[y - 1, x]
    p3 = img01[y - 1, x + 1]
    p4 = img01[y, x + 1]
    p5 = img01[y + 1, x + 1]
    p6 = img01[y + 1, x]
    p7 = img01[y + 1, x - 1]
    p8 = img01[y, x - 1]
    p9 = img01[y - 1, x - 1]

    seq = [p2, p3, p4, p5, p6, p7, p8, p9, p2]  # circular
    # total number of changes 0<->1
    return int(sum(seq[i] != seq[i + 1] for i in range(8)))


def calculate_q9(
    R_mask: np.ndarray,
    vein_img_or_path: Union[str, np.ndarray],
    Fc: int = 15,
    *,
    min_area: int = 120,
    prune_iters: int = 30,
    min_skel_len: int = 120,
    keep_largest: bool = True,
):
    if R_mask is None:
        return 0, 0, 0, 0, np.zeros((1, 1), dtype=np.uint8)

    fg = (R_mask == 255)
    if np.count_nonzero(fg) == 0:
        return 0, 0, 0, 0, np.zeros_like(R_mask, dtype=np.uint8)
    
    _, skel01 = prepare_vessel_skeleton(
        R_mask,
        vein_img_or_path,
        min_area=min_area,
        prune_iters=prune_iters,
        min_skel_len=min_skel_len,
        keep_largest=keep_largest,
    )


    # --- TOPOLOGY ---
    H, W = skel01.shape
    N_end, N_int = 0, 0

    for y in range(1, H - 1):
        for x in range(1, W - 1):
            if skel01[y, x] != 1:
                continue
            T = _transition_count_8nbr(skel01, y, x)
            if T == 2:
                N_end += 1
            elif T == 6 or T == 8:
                N_int += 1

    N_fp = N_end + N_int
    Q9 = min(100, int(round((N_fp / Fc) * 100))) if Fc > 0 else 0

    return Q9, N_fp, N_end, N_int, skel01



