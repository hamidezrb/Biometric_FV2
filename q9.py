"""

ISO/IEC 29794-9 Quality Component 9 (Number of Feature Points) — Clause 5.2.9.

"""



from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np



from iso_constants import CaptureSite, DEFAULT_CAPTURE_SITE, get_capture_site_coefficients

from iso_foreground import is_foreground_region_valid

from vessel_utils import prepare_vessel_skeleton



# Clockwise 8-neighbour offsets from center p: p2..p9 (top, top-right, …, top-left).

_CLOCKWISE_OFFSETS = (

    (-1, 0),

    (-1, 1),

    (0, 1),

    (1, 1),

    (1, 0),

    (1, -1),

    (0, -1),

    (-1, -1),

)





def _neighbor_values_clockwise(

    img01: np.ndarray,

    y: int,

    x: int,

) -> Tuple[int, ...]:

    """

    Eight neighbours p2..p9 clockwise; pixels outside the image are 0.

    """

    h, w = img01.shape

    out = []

    for dy, dx in _CLOCKWISE_OFFSETS:

        ny, nx = y + dy, x + dx

        if 0 <= ny < h and 0 <= nx < w:

            out.append(int(img01[ny, nx] == 1))

        else:

            out.append(0)

    return tuple(out)





def transition_count_8nbr(img01: np.ndarray, y: int, x: int) -> int:

    """

    Count 0↔1 transitions around the 8-neighbour ring (clockwise, Clause 5.2.9 c).

    """

    ring = _neighbor_values_clockwise(img01, y, x)

    seq = list(ring) + [ring[0]]

    return int(sum(seq[i] != seq[i + 1] for i in range(8)))





def count_feature_points_in_foreground(

    skel01: np.ndarray,

    foreground_mask: np.ndarray,

) -> Tuple[int, int, int]:

    """

    Traverse left→right, top→bottom; count endpoints (T=2) and intersections (T=6,8).



    Returns (N_endpoint, N_intersection, N_fp).

    Only pixels with value 1 inside the foreground mask are evaluated.

    """

    if skel01.shape != foreground_mask.shape:

        raise ValueError("skeleton and foreground mask shape mismatch")



    h, w = skel01.shape

    n_end, n_int = 0, 0

    for y in range(h):

        for x in range(w):

            if not foreground_mask[y, x] or skel01[y, x] != 1:

                continue

            t = transition_count_8nbr(skel01, y, x)

            if t == 2:

                n_end += 1

            elif t in (6, 8):

                n_int += 1



    return n_end, n_int, n_end + n_int





def compute_q9_score(N_fp: int, Fc: int) -> int:

    """

    Formula (16) — map feature-point count to [0, 100] and round.



    Q9 = MIN(100, ROUND((N_ep + N_cp) / Pc * 100))

    """

    if Fc <= 0 or N_fp < 0:

        return 0

    q9_raw = (float(N_fp) / float(Fc)) * 100.0

    return max(0, min(100, int(round(q9_raw))))





def calculate_q9(

    R_mask: np.ndarray,

    vein_img_or_path: Union[str, np.ndarray],

    capture_site: CaptureSite = DEFAULT_CAPTURE_SITE,

    Fc: Optional[int] = None,

    S_unoccluded: Optional[int] = None,

    *,

    iso_minimal: bool = False,

) -> Tuple[int, int, int, int, np.ndarray]:

    """

    ISO/IEC 29794-9 Clause 5.2.9 — number of feature points Q9.



    Step a): foreground from R_mask; invalid => Q9 = 0.

    Steps b–c): binarize within R, thin, count endpoints/intersections in R.

    Steps d–e): Table 3 Pc (Fc), Formula (16).

    """

    if R_mask is None:

        return 0, 0, 0, 0, np.zeros((1, 1), dtype=np.uint8)



    fg = (R_mask == 255)

    if S_unoccluded is None:

        S_unoccluded = int(np.count_nonzero(fg))



    if not is_foreground_region_valid(S_unoccluded):

        return 0, 0, 0, 0, np.zeros_like(R_mask, dtype=np.uint8)



    _, skel01 = prepare_vessel_skeleton(

        R_mask, vein_img_or_path, iso_minimal=iso_minimal

    )



    N_end, N_int, N_fp = count_feature_points_in_foreground(skel01, fg)

    effective_fc = (

        Fc if Fc is not None

        else get_capture_site_coefficients(capture_site)["FC"]

    )



    Q9 = compute_q9_score(N_fp, effective_fc)

    return Q9, N_fp, N_end, N_int, skel01


