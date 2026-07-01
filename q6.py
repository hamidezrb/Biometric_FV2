"""
ISO/IEC 29794-9 Quality Component 6 (Sharpness) — Clause 5.2.6.
"""

import cv2
import numpy as np
from typing import List, NamedTuple, Optional, Tuple

from iso_constants import GC_SHARPNESS
from iso_foreground import is_foreground_region_valid


class Q6Result(NamedTuple):
    """Detailed Q6 outputs for audit, logging, and debug visualization."""

    Q6_score: int
    N100: int
    q6_raw: float
    S_unoccluded: int
    gc: float
    foreground_pixel_count: int
    edge_pixel_count: int
    edge_pixel_count_fg: int
    w_mean: np.ndarray
    w_norm: np.ndarray
    edge_mask: np.ndarray

# Figure 1 — Sobel operators (PWI draft 29794-9), exactly as in Clause 5.2.6.2.
ISO_SOBEL_KERNELS: Tuple[np.ndarray, ...] = (
    np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32),
    np.array([[2, 1, 0], [1, 0, -1], [0, -1, -2]], dtype=np.float32),
    np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32),
    np.array([[0, -1, -2], [1, 0, -1], [2, 1, 0]], dtype=np.float32),
)

N100_THRESHOLD = 100


def _convolve_iso_sobel_operators(gray_fg: np.ndarray) -> np.ndarray:
    """
    Steps c–d: convolve Figure 1 kernels (zero border), then Formula (11).

    W_mean = sum(|I_i|) / 4, i = 1..4
    """
    gray_f = gray_fg.astype(np.float32)
    convolved: List[np.ndarray] = []
    for kernel in ISO_SOBEL_KERNELS:
        I_i = cv2.filter2D(
            gray_f, cv2.CV_32F, kernel, borderType=cv2.BORDER_CONSTANT
        )
        convolved.append(I_i)
    return sum(np.abs(I_i) for I_i in convolved) / 4.0


def _minmax_normalize_to_uint8(response: np.ndarray) -> np.ndarray:
    """Step e: min-max normalize W_mean to [0, 255]."""
    mn = float(np.min(response))
    mx = float(np.max(response))
    if mx <= mn:
        return np.zeros(response.shape, dtype=np.uint8)
    scaled = (response - mn) / (mx - mn) * 255.0
    return scaled.astype(np.uint8)


def compute_q6_raw(
    N100: int,
    S_unoccluded: int,
    gc: float = GC_SHARPNESS,
) -> float:
    """Formula (12) value before rounding and capping."""
    if not is_foreground_region_valid(S_unoccluded) or gc <= 0:
        return 0.0
    return (N100 / (gc * float(S_unoccluded))) * 100.0


def compute_q6_score(
    N100: int,
    S_unoccluded: int,
    gc: float = GC_SHARPNESS,
) -> int:
    """
    Formula (12) — map N100 to [0, 100] and round.

    Q6 = MIN(100, ROUND(N100 / (gc * S_unoccluded) * 100))

    The draft page shows the ratio form; the *100 scale factor matches
    Table-style quality mapping used elsewhere in ISO/IEC 29794-9 (cf. Q1).
    """
    q6_raw = compute_q6_raw(N100, S_unoccluded, gc=gc)
    return max(0, min(100, int(round(q6_raw))))


def compute_q6_intermediates(
    R_mask: np.ndarray,
    Grayscale_Image: np.ndarray,
    *,
    threshold: int = N100_THRESHOLD,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int]:
    """
    ISO steps c–f: masked convolution, min-max normalization, edge-pixel count.

    Returns:
        w_mean, w_norm, edge_mask, foreground_pixel_count,
        edge_pixel_count (full image), edge_pixel_count_fg
    """
    fg = (R_mask == 255)
    foreground_pixel_count = int(np.count_nonzero(fg))

    gray = Grayscale_Image
    if gray.dtype != np.uint8:
        gray = gray.astype(np.uint8)

    gray_fg = gray.copy()
    gray_fg[~fg] = 0

    w_mean = _convolve_iso_sobel_operators(gray_fg)
    w_norm = _minmax_normalize_to_uint8(w_mean)
    edge_mask = w_norm > threshold
    edge_pixel_count = int(np.count_nonzero(edge_mask))
    edge_pixel_count_fg = int(np.count_nonzero(edge_mask & fg))
    return (
        w_mean,
        w_norm,
        edge_mask,
        foreground_pixel_count,
        edge_pixel_count,
        edge_pixel_count_fg,
    )


def calculate_q6_detailed(
    R_mask: np.ndarray,
    Grayscale_Image: np.ndarray,
    S_unoccluded: Optional[int] = None,
    gc: float = GC_SHARPNESS,
    threshold: int = N100_THRESHOLD,
) -> Q6Result:
    """ISO/IEC 29794-9 Clause 5.2.6 with full audit outputs."""
    fg = (R_mask == 255)
    if S_unoccluded is None:
        S_unoccluded = int(np.count_nonzero(fg))

    empty = np.zeros_like(Grayscale_Image, dtype=np.uint8)
    if not is_foreground_region_valid(S_unoccluded):
        return Q6Result(
            Q6_score=0,
            N100=0,
            q6_raw=0.0,
            S_unoccluded=S_unoccluded,
            gc=gc,
            foreground_pixel_count=int(np.count_nonzero(fg)),
            edge_pixel_count=0,
            edge_pixel_count_fg=0,
            w_mean=empty.astype(np.float32),
            w_norm=empty,
            edge_mask=empty.astype(bool),
        )

    (
        w_mean,
        w_norm,
        edge_mask,
        foreground_pixel_count,
        edge_pixel_count,
        edge_pixel_count_fg,
    ) = compute_q6_intermediates(
        R_mask,
        Grayscale_Image,
        threshold=threshold,
    )
    N100 = edge_pixel_count
    q6_raw = compute_q6_raw(N100, S_unoccluded, gc=gc)
    Q6 = compute_q6_score(N100, S_unoccluded, gc=gc)
    return Q6Result(
        Q6_score=Q6,
        N100=N100,
        q6_raw=q6_raw,
        S_unoccluded=S_unoccluded,
        gc=gc,
        foreground_pixel_count=foreground_pixel_count,
        edge_pixel_count=edge_pixel_count,
        edge_pixel_count_fg=edge_pixel_count_fg,
        w_mean=w_mean,
        w_norm=w_norm,
        edge_mask=edge_mask,
    )


def calculate_q6(
    R_mask: np.ndarray,
    Grayscale_Image: np.ndarray,
    S_unoccluded: Optional[int] = None,
    gc: float = GC_SHARPNESS,
    threshold: int = N100_THRESHOLD,
) -> Tuple[int, int]:
    """
    ISO/IEC 29794-9 Clause 5.2.6 — sharpness Q6.

    Step a): foreground from R_mask; invalid => Q6 = 0.
    Steps b–g): ISO Sobel kernels, Formula (11), min-max, N100, gc.
    Step h): Formula (12) via compute_q6_score.
    """
    fg = (R_mask == 255)
    if S_unoccluded is None:
        S_unoccluded = int(np.count_nonzero(fg))

    result = calculate_q6_detailed(
        R_mask,
        Grayscale_Image,
        S_unoccluded=S_unoccluded,
        gc=gc,
        threshold=threshold,
    )
    return result.Q6_score, result.N100


def build_q6_debug_images(
    gray: np.ndarray,
    r_mask: np.ndarray,
    q6_result: Q6Result,
) -> dict[str, np.ndarray]:
    """Debug panels for Q6: original, foreground, Sobel response, edge mask."""
    original = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    fg_vis = original.copy()
    fg = r_mask == 255
    if np.any(fg):
        fg_vis[fg] = (0.6 * fg_vis[fg] + 0.4 * np.array((0, 255, 0))).astype(np.uint8)

    sob_vis = cv2.applyColorMap(q6_result.w_norm, cv2.COLORMAP_INFERNO)
    edge_vis = original.copy()
    edge = q6_result.edge_mask
    if np.any(edge):
        edge_vis[edge] = (0.5 * edge_vis[edge] + 0.5 * np.array((0, 0, 255))).astype(
            np.uint8
        )

    return {
        "q6_original.png": original,
        "q6_foreground_mask.png": fg_vis,
        "q6_sobel_response.png": sob_vis,
        "q6_edge_pixels.png": edge_vis,
    }
