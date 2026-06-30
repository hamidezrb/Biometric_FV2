from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Tuple, Union
import cv2
import numpy as np


@dataclass(frozen=True)
class OpenVeinVesselCleanupConfig:
    """
    Optional pre-thinning cleanup for OpenVein MATLAB vein maps (Q8/Q9 input).

    ISO/IEC 29794-9 Clause 5.2.8/5.2.9 normative steps are only:
      b) binarize the vessel map within R
      c) thin to a 1-pixel skeleton and count pixels / feature points

    All boolean flags below are **non-normative heuristics**. They can reduce
    N_vessel (Q8) and N_fp / endpoints / intersections (Q9) by removing noise,
    filling holes, and pruning spurs — use ``iso_minimal()`` for strict ISO
  reproduction on OpenVein exports.
    """

    remove_small_components: bool = False
    min_area: int | None = None

    remove_border_touching: bool = False

    roi_margin: int = 0

    fill_small_holes: bool = False
    hole_max_area: int = 100

    morphological_opening: bool = False
    open_ksize: int = 5

    prune_spurs: bool = False
    prune_iters: int = 30

    remove_short_skeleton_branches: bool = False
    min_skel_len: int = 40

    @property
    def preset_name(self) -> str:
        if self == OpenVeinVesselCleanupConfig.iso_minimal():
            return "iso_minimal"
        if self == OpenVeinVesselCleanupConfig.heuristic_default():
            return "heuristic_default"
        return "custom"

    @property
    def is_iso_minimal(self) -> bool:
        return self == OpenVeinVesselCleanupConfig.iso_minimal()

    def summary(self) -> str:
        return f"{self.preset_name}({', '.join(f'{k}={v}' for k, v in asdict(self).items())})"

    @classmethod
    def iso_minimal(cls) -> OpenVeinVesselCleanupConfig:
        """Normative steps b–c only: binarize within R, thin once."""
        return cls()

    @classmethod
    def heuristic_default(cls) -> OpenVeinVesselCleanupConfig:
        """Project default for noisy OpenVein PC maps (not ISO-defined)."""
        return cls(
            remove_small_components=True,
            remove_border_touching=False,
            roi_margin=3,
            fill_small_holes=True,
            hole_max_area=100,
            morphological_opening=True,
            open_ksize=5,
            prune_spurs=True,
            prune_iters=30,
            remove_short_skeleton_branches=True,
            min_skel_len=40,
        )


DEFAULT_OPENVEIN_VESSEL_CLEANUP = OpenVeinVesselCleanupConfig.heuristic_default()


def vessel_cleanup_from_preset(name: str) -> OpenVeinVesselCleanupConfig:
    presets = {
        "iso_minimal": OpenVeinVesselCleanupConfig.iso_minimal(),
        "heuristic_default": OpenVeinVesselCleanupConfig.heuristic_default(),
    }
    key = name.strip().lower()
    if key not in presets:
        raise ValueError(
            f"Unknown vessel cleanup preset {name!r}. Choose from: {', '.join(presets)}."
        )
    return presets[key]


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


def is_openvein_binary_map(vessel: np.ndarray) -> bool:
    """True when the map is already OpenVein-style binary (0 / 255 only)."""
    if vessel.ndim != 2:
        return False
    vals = np.unique(vessel)
    return len(vals) <= 2 and set(int(v) for v in vals.tolist()).issubset({0, 255})


def binarize_openvein_vein_map(
    vessel: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """
    Binarize an OpenVein MATLAB export (uint8 0/255) within a foreground mask.

    ``mask`` may be uint8 (255 = foreground) or a boolean foreground array.
    Uses ``>= 128`` so already-binary maps are not re-thresholded aggressively.
    """
    if mask.dtype == np.bool_:
        fg = mask
    else:
        fg = mask == 255
    if is_openvein_binary_map(vessel):
        return ((vessel >= 128) & fg).astype(np.uint8)
    # Grayscale / response map fallback (non-MATLAB backends).
    _, thr = cv2.threshold(vessel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return ((thr > 0) & fg).astype(np.uint8)


def remove_border_touching_components(binary01: np.ndarray) -> np.ndarray:
    """Drop connected components that touch the image border."""
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary01.astype(np.uint8), connectivity=8
    )
    if num <= 1:
        return binary01.astype(np.uint8)
    h, w = binary01.shape
    out = np.zeros_like(binary01, dtype=np.uint8)
    for k in range(1, num):
        x, y, bw, bh, _area = stats[k]
        if x <= 0 or y <= 0 or (x + bw) >= w or (y + bh) >= h:
            continue
        out[labels == k] = 1
    return out


def fill_small_holes(binary01: np.ndarray, max_area: int) -> np.ndarray:
    """Fill enclosed background holes smaller than ``max_area`` pixels."""
    if max_area <= 0:
        return binary01.astype(np.uint8)
    inv = (1 - binary01).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    out = binary01.astype(np.uint8).copy()
    h, w = binary01.shape
    for k in range(1, num):
        x, y, bw, bh, area = stats[k]
        if x <= 0 or y <= 0 or (x + bw) >= w or (y + bh) >= h:
            continue
        if area <= max_area:
            out[labels == k] = 1
    return out


def clip_to_roi_interior(
    binary01: np.ndarray,
    mask: np.ndarray,
    *,
    margin: int = 3,
) -> np.ndarray:
    """Keep vein pixels at least ``margin`` pixels inside the ROI boundary."""
    if margin <= 0:
        return binary01.astype(np.uint8)
    if mask.dtype == np.bool_:
        mask_u8 = mask.astype(np.uint8)
    else:
        mask_u8 = (mask == 255).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    inner = cv2.erode(mask_u8, kernel, iterations=margin)
    return (binary01.astype(np.uint8) & inner).astype(np.uint8)


def _prepare_vessel_skeleton_openvein(
    R_mask: np.ndarray,
    vein_img_or_path: Union[str, np.ndarray],
    *,
    R_unoccluded: np.ndarray | None = None,
    cleanup: OpenVeinVesselCleanupConfig | None = None,
) -> Tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """
    Prepare a 1-pixel skeleton from an OpenVein MATLAB uint8 0/255 vein map.

    When ``cleanup.is_iso_minimal`` (preset ``iso_minimal``): Clause 5.2.8/5.2.9
    steps b–c — binarize within the unoccluded foreground, Zhang–Suen thin once.

    Otherwise applies documented heuristics (speckle/border/ROI cleanup, hole
    fill, opening, spur pruning) before thinning; see ``OpenVeinVesselCleanupConfig``.
    """
    cfg = cleanup or DEFAULT_OPENVEIN_VESSEL_CLEANUP
    fg = (R_mask == 255)
    mask_for_bin = (R_unoccluded == 255) if R_unoccluded is not None else fg
    vessel = load_gray(vein_img_or_path)
    if vessel.shape != R_mask.shape:
        raise ValueError(f"Shape mismatch: vessel={vessel.shape} vs R_mask={R_mask.shape}")

    before_clean = binarize_openvein_vein_map(vessel, mask_for_bin)
    cleaned = before_clean.copy()

    if not cfg.is_iso_minimal:
        min_area = cfg.min_area
        if min_area is None:
            min_area = max(50, int(0.00025 * int(np.count_nonzero(mask_for_bin))))

        if cfg.remove_small_components:
            cleaned = remove_small_components(cleaned, min_area=min_area)
        if cfg.remove_border_touching:
            cleaned = remove_border_touching_components(cleaned)
        if cfg.roi_margin > 0:
            cleaned = clip_to_roi_interior(cleaned, mask_for_bin, margin=cfg.roi_margin)
        if cfg.fill_small_holes:
            cleaned = fill_small_holes(cleaned, cfg.hole_max_area)
        if cfg.morphological_opening and cfg.open_ksize > 1:
            kernel = np.ones((cfg.open_ksize, cfg.open_ksize), np.uint8)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

    skel01 = zs_thinning(cleaned)
    if not cfg.is_iso_minimal:
        if cfg.prune_spurs and cfg.prune_iters > 0:
            skel01 = prune_spurs(skel01, iterations=cfg.prune_iters)
        if cfg.remove_short_skeleton_branches and cfg.min_skel_len > 1:
            skel01 = remove_short_skeleton_components(skel01, min_len=cfg.min_skel_len)

    stages = {
        "pc_feature_raw": vessel,
        "pc_binary_before_cleaning": (before_clean * 255).astype(np.uint8),
        "pc_binary_after_cleaning": (cleaned * 255).astype(np.uint8),
        "pc_skeleton": (skel01 * 255).astype(np.uint8),
    }
    return cleaned, skel01, stages


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
    vein_map_source: str = "iso",
    R_unoccluded: np.ndarray | None = None,
    openvein_cleanup: OpenVeinVesselCleanupConfig | None = None,
    min_area: int | None = None,
    close_ksize: int = 3,
    keep_largest: bool = True,
    prune_iters: int = 30,
    min_skel_len: int = 120,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (vessel01, skel01), both 0/1 uint8 aligned to R_mask.

    vein_map_source:
      - ``iso``: Clause 5.2.8/5.2.9 — binarize within R and thin only.
      - ``openvein_matlab``: OpenVein uint8 0/255 export; cleanup controlled by
        ``openvein_cleanup`` (default: heuristic_default; use iso_minimal() for
        normative steps b–c only).
    """
    if vein_map_source == "openvein_matlab":
        vessel01, skel01, _stages = _prepare_vessel_skeleton_openvein(
            R_mask,
            vein_img_or_path,
            R_unoccluded=R_unoccluded,
            cleanup=openvein_cleanup,
        )
        return vessel01, skel01
    if iso_minimal:
        return _prepare_vessel_skeleton_iso(R_mask, vein_img_or_path)
    return _prepare_vessel_skeleton_advanced(
        R_mask,
        vein_img_or_path,
        min_area=min_area or 120,
        close_ksize=close_ksize,
        keep_largest=keep_largest,
        prune_iters=prune_iters,
        min_skel_len=min_skel_len,
    )


def prepare_vessel_skeleton_with_stages(
    R_mask: np.ndarray,
    vein_img_or_path: Union[str, np.ndarray],
    *,
    vein_map_source: str = "openvein_matlab",
    R_unoccluded: np.ndarray | None = None,
    openvein_cleanup: OpenVeinVesselCleanupConfig | None = None,
) -> Tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Like prepare_vessel_skeleton but also returns intermediate debug images."""
    if vein_map_source == "openvein_matlab":
        return _prepare_vessel_skeleton_openvein(
            R_mask,
            vein_img_or_path,
            R_unoccluded=R_unoccluded,
            cleanup=openvein_cleanup,
        )
    vessel01, skel01 = prepare_vessel_skeleton(
        R_mask,
        vein_img_or_path,
        vein_map_source="iso",
    )
    vessel = load_gray(vein_img_or_path)
    return vessel01, skel01, {
        "pc_feature_raw": vessel,
        "pc_binary_before_cleaning": (vessel01 * 255).astype(np.uint8),
        "pc_binary_after_cleaning": (vessel01 * 255).astype(np.uint8),
        "pc_skeleton": (skel01 * 255).astype(np.uint8),
    }
