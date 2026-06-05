from __future__ import annotations
from typing import Tuple, Union
import cv2
import numpy as np


def load_gray(img_or_path: Union[str, np.ndarray]) -> np.ndarray:
    if isinstance(img_or_path, str):
        img = cv2.imread(img_or_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read image: {img_or_path}")
        return img
    if isinstance(img_or_path, np.ndarray):
        return img_or_path
    raise TypeError("img_or_path must be a file path or a numpy array.")


def keep_largest_component(binary01: np.ndarray) -> np.ndarray:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary01.astype(np.uint8), connectivity=8)
    if num <= 1:
        return binary01.astype(np.uint8)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == largest).astype(np.uint8)


def remove_small_components(binary01: np.ndarray, min_area: int) -> np.ndarray:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary01.astype(np.uint8), connectivity=8)
    out = np.zeros_like(binary01, dtype=np.uint8)
    for k in range(1, num):
        if stats[k, cv2.CC_STAT_AREA] >= min_area:
            out[labels == k] = 1
    return out


def zs_thinning(binary01: np.ndarray) -> np.ndarray:
    """
    Zhang–Suen thinning (0/1 uint8 in and out).

    ISO/IEC 29794-9 Clause 5.2.8 Step b) cites thinning per reference [5].
    TODO: Confirm normative [5] in the published standard matches Zhang–Suen;
    this implementation is used for both Q8 and Q9 when iso_minimal=True.
    """
    img = (binary01 > 0).astype(np.uint8)
    if img.size == 0:
        return img

    changed = True
    H, W = img.shape

    def nbrs(y: int, x: int) -> Tuple[int, int, int, int, int, int, int, int]:
        p2 = img[y - 1, x]
        p3 = img[y - 1, x + 1]
        p4 = img[y, x + 1]
        p5 = img[y + 1, x + 1]
        p6 = img[y + 1, x]
        p7 = img[y + 1, x - 1]
        p8 = img[y, x - 1]
        p9 = img[y - 1, x - 1]
        return p2, p3, p4, p5, p6, p7, p8, p9

    while changed:
        changed = False
        to_remove = []

        for y in range(1, H - 1):
            for x in range(1, W - 1):
                if img[y, x] != 1:
                    continue
                p2, p3, p4, p5, p6, p7, p8, p9 = nbrs(y, x)
                B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
                if B < 2 or B > 6:
                    continue
                seq = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
                A = sum((seq[i] == 0 and seq[i + 1] == 1) for i in range(8))
                if A != 1:
                    continue
                if p2 * p4 * p6 != 0:
                    continue
                if p4 * p6 * p8 != 0:
                    continue
                to_remove.append((y, x))

        if to_remove:
            for y, x in to_remove:
                img[y, x] = 0
            changed = True

        to_remove = []
        for y in range(1, H - 1):
            for x in range(1, W - 1):
                if img[y, x] != 1:
                    continue
                p2, p3, p4, p5, p6, p7, p8, p9 = nbrs(y, x)
                B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
                if B < 2 or B > 6:
                    continue
                seq = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
                A = sum((seq[i] == 0 and seq[i + 1] == 1) for i in range(8))
                if A != 1:
                    continue
                if p2 * p4 * p8 != 0:
                    continue
                if p2 * p6 * p8 != 0:
                    continue
                to_remove.append((y, x))

        if to_remove:
            for y, x in to_remove:
                img[y, x] = 0
            changed = True

    return img.astype(np.uint8)


def prune_spurs(skel01: np.ndarray, iterations: int = 10) -> np.ndarray:
    skel = skel01.copy().astype(np.uint8)
    for _ in range(iterations):
        nbr = (
            skel[:-2, :-2] + skel[:-2, 1:-1] + skel[:-2, 2:] +
            skel[1:-1, :-2]                 + skel[1:-1, 2:] +
            skel[2:, :-2] + skel[2:, 1:-1] + skel[2:, 2:]
        )
        center = skel[1:-1, 1:-1]
        endpoints = (center == 1) & (nbr == 1)
        if not np.any(endpoints):
            break
        skel[1:-1, 1:-1][endpoints] = 0
    return skel


def remove_short_skeleton_components(skel01: np.ndarray, min_len: int) -> np.ndarray:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(skel01.astype(np.uint8), connectivity=8)
    out = np.zeros_like(skel01, dtype=np.uint8)
    for k in range(1, num):
        if stats[k, cv2.CC_STAT_AREA] >= min_len:
            out[labels == k] = 1
    return out


def _prepare_vessel_skeleton_iso(
    R_mask: np.ndarray,
    vein_img_or_path: Union[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    ISO Clause 5.2.8/5.2.9 Steps b–c: binarize vessels within R, thin to 1 px.

    Vessel map is binarized as (vein > 0) ∧ R, stored as 0/1 (not 255).
    TODO: The draft does not define how to derive the vessel map from the
    vascular image; callers may supply an external binarized vein map.
    """
    fg = (R_mask == 255)
    vessel = load_gray(vein_img_or_path)
    if vessel.shape != R_mask.shape:
        raise ValueError(f"Shape mismatch: vessel={vessel.shape} vs R_mask={R_mask.shape}")

    vessel01 = ((vessel > 0) & fg).astype(np.uint8)
    skel01 = zs_thinning(vessel01)
    return vessel01, skel01


def _prepare_vessel_skeleton_advanced(
    R_mask: np.ndarray,
    vein_img_or_path: Union[str, np.ndarray],
    *,
    min_area: int = 120,
    close_ksize: int = 3,
    keep_largest: bool = True,
    prune_iters: int = 30,
    min_skel_len: int = 120,
) -> Tuple[np.ndarray, np.ndarray]:
    """Non-ISO optional cleanup (disabled by default in the public API)."""
    fg = (R_mask == 255)
    vessel = load_gray(vein_img_or_path)
    if vessel.shape != R_mask.shape:
        raise ValueError(f"Shape mismatch: vessel={vessel.shape} vs R_mask={R_mask.shape}")

    v = ((vessel > 0) & fg).astype(np.uint8)
    clean = remove_small_components(v, min_area=min_area)
    if close_ksize and close_ksize > 1:
        kernel = np.ones((close_ksize, close_ksize), np.uint8)
        clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)
    vessel01 = keep_largest_component(clean) if keep_largest else clean
    skel01 = zs_thinning(vessel01)
    if prune_iters and prune_iters > 0:
        skel01 = prune_spurs(skel01, iterations=prune_iters)
    if min_skel_len and min_skel_len > 1:
        skel01 = remove_short_skeleton_components(skel01, min_len=min_skel_len)
    return vessel01, skel01


def prepare_vessel_skeleton(
    R_mask: np.ndarray,
    vein_img_or_path: Union[str, np.ndarray],
    *,
    iso_minimal: bool = True,
    min_area: int = 120,
    close_ksize: int = 3,
    keep_largest: bool = True,
    prune_iters: int = 30,
    min_skel_len: int = 120,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (vessel01, skel01), both 0/1 uint8 aligned to R_mask.

    iso_minimal=True (default): ISO path — binarize within R and thin only.
    iso_minimal=False: optional non-ISO morphology/pruning (debug only).
    """
    if iso_minimal:
        return _prepare_vessel_skeleton_iso(R_mask, vein_img_or_path)
    return _prepare_vessel_skeleton_advanced(
        R_mask,
        vein_img_or_path,
        min_area=min_area,
        close_ksize=close_ksize,
        keep_largest=keep_largest,
        prune_iters=prune_iters,
        min_skel_len=min_skel_len,
    )
