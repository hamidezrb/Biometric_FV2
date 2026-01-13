import numpy as np
import cv2
from typing import Tuple, Optional


# ---------------------------
# Pc selection (Table 3)
# ---------------------------
def get_pc(capture_site: str = "finger", includes_fingers: bool = False) -> int:
    """
    PWI Table 3: coefficient Pc for feature point count mapping.

    capture_site:
      - "finger"  -> Pc = 15
      - "palm"    -> Pc = 25 (palm ventral surface or back of the hand)
      - "dorsal"  -> Pc = 25 (if only dorsal hand area)
      - "entire"  -> Pc = 50 (entire palm/dorsal including fingers)
    includes_fingers: if True -> Pc = 50 for palm/dorsal
    """
    s = capture_site.strip().lower()
    if s == "finger":
        return 15
    if s in ("palm", "dorsal", "hand"):
        return 50 if includes_fingers else 25
    if s in ("entire", "full"):
        return 50
    # safe default for PALMAR datasets (usually palm area): 25
    return 25


# ---------------------------
# Skeletonization
# ---------------------------
def skeletonize_binary(binary_img: np.ndarray) -> np.ndarray:
    """
    Skeletonize a binary image into 1-pixel skeleton.
    Input: uint8/bool, vessels>0
    Output: uint8 0/1
    """
    b = (binary_img > 0).astype(np.uint8)

    # Prefer OpenCV contrib thinning if available
    try:
        skel = cv2.ximgproc.thinning(b * 255, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
        return (skel > 0).astype(np.uint8)
    except Exception:
        pass

    # Fallback: simple morphological skeleton (no extra dependencies)
    size = np.size(b)
    skel = np.zeros(b.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    img = b.copy() * 255

    while True:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        if size - cv2.countNonZero(img) == size:
            break

    return (skel > 0).astype(np.uint8)


# ---------------------------
# Pruning (to avoid Q9 saturation)
# ---------------------------
def prune_skeleton_components(skel01: np.ndarray, min_len: int = 25) -> np.ndarray:
    """
    Remove tiny skeleton components (very short broken segments),
    which otherwise create many endpoints and saturate Q9.

    Input/output: uint8 0/1
    """
    if min_len <= 1:
        return skel01

    num, labels, stats, _ = cv2.connectedComponentsWithStats(skel01.astype(np.uint8), connectivity=8)
    out = np.zeros_like(skel01, dtype=np.uint8)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_len:
            out[labels == i] = 1
    return out


# ---------------------------
# Transition counting (PWI rule)
# ---------------------------
def transitions_0_1_total(neigh01: np.ndarray) -> int:
    """
    neigh01: 8 neighbors in clockwise order:
      p2 p3 p4
      p9  c p5
      p8 p7 p6

    PWI text: "number of transitions between 0 and 1 among these pixel values"
    => count total changes (0->1 OR 1->0) around the ring.

    That yields:
      endpoint -> 2 transitions
      intersection -> 6 or 8 transitions
    """
    n = neigh01.astype(np.uint8)
    seq = np.array([n[0], n[1], n[2], n[3], n[4], n[5], n[6], n[7], n[0]], dtype=np.uint8)
    # count changes between consecutive values
    return int(np.sum(seq[:-1] != seq[1:]))


# ---------------------------
# Main Q9 function (PWI 5.2.9)
# ---------------------------
def calculate_q9(
    vessel_binary: np.ndarray,
    foreground_mask: np.ndarray,
    capture_site: str = "palm",
    includes_fingers: bool = False,
    min_skel_component_len: int = 25,
) -> Tuple[int, int, int]:
    """
    PWI 5.2.9 Number of feature points in vascular image quality.

    Steps (matches your screenshots):
    a) Extract foreground (if invalid -> Q9=0)
    b) Binarize + thin (skeleton thickness 1 pixel)
    c) For each skeleton pixel p=1:
       check 8 neighbors in clockwise direction.
       if transitions(0<->1) == 2 -> endpoint
       if transitions(0<->1) == 6 or 8 -> intersection
    d) Choose Pc from Table 3
    e) Q9 = min(100, round((Nep + Ncp)/Pc * 100))

    Returns: (Q9, Nep, Ncp)
    """
    if vessel_binary.shape != foreground_mask.shape:
        raise ValueError("vessel_binary and foreground_mask must have same shape")

    fg = (foreground_mask > 0)
    if np.count_nonzero(fg) == 0:
        return 0, 0, 0

    # Apply foreground mask to vessel binary
    vb = np.zeros_like(vessel_binary, dtype=np.uint8)
    vb[fg] = (vessel_binary[fg] > 0).astype(np.uint8) * 255

    # Thin to skeleton (0/1)
    skel = skeletonize_binary(vb)
    skel = (skel > 0).astype(np.uint8)
    skel[~fg] = 0

    # Prune tiny skeleton components to avoid endpoint explosion
    skel = prune_skeleton_components(skel, min_len=min_skel_component_len)

    # Count endpoints and intersections
    padded = np.pad(skel, 1, mode="constant", constant_values=0)
    ys, xs = np.where(skel == 1)

    Nep = 0
    Ncp = 0

    for y, x in zip(ys, xs):
        yy, xx = y + 1, x + 1
        block = padded[yy-1:yy+2, xx-1:xx+2]

        neigh = np.array([
            block[0, 1],  # p2 (top)
            block[0, 2],  # p3 (top-right)
            block[1, 2],  # p4 (right)
            block[2, 2],  # p5 (bottom-right)
            block[2, 1],  # p6 (bottom)
            block[2, 0],  # p7 (bottom-left)
            block[1, 0],  # p8 (left)
            block[0, 0],  # p9 (top-left)
        ], dtype=np.uint8)

        T = transitions_0_1_total(neigh)

        if T == 2:
            Nep += 1
        elif T == 6 or T == 8:
            Ncp += 1

    Pc = get_pc(capture_site=capture_site, includes_fingers=includes_fingers)

    Q9 = int(round(((Nep + Ncp) / float(Pc)) * 100.0)) if Pc > 0 else 0
    Q9 = max(0, min(100, Q9))

    return Q9, Nep, Ncp
