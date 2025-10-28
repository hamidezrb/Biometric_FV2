import cv2
import numpy as np
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from typing import Tuple, Optional


def calculate_q2(image_path: str) -> Tuple[int, Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    Q2 — Offset Complement (Centering) using libraries.
    Steps:
      1. Read image as grayscale.
      2. Otsu threshold (both dark and bright masks).
      3. Extract largest connected component.
      4. Compute centroid (cx, cy) using regionprops.
      5. Measure distance from image center.
      6. Normalize and map to [0,100].
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"[Q2] ERROR: Cannot read {image_path}")
        return 0, None, None, None, None

    H, W = img.shape
    x0, y0 = (W - 1) / 2.0, (H - 1) / 2.0

    try:
        T = threshold_otsu(img)
    except Exception:
        T = np.median(img)

    mask_bright = (img >= T).astype(np.uint8)
    mask_dark = (img <= T).astype(np.uint8)

    def largest_component(mask):
        lab = label(mask, connectivity=2)
        if lab.max() == 0:
            return np.zeros_like(mask)
        areas = [(lab == k).sum() for k in range(1, lab.max() + 1)]
        k_star = 1 + int(np.argmax(areas))
        return (lab == k_star).astype(np.uint8)

    fg_bright = largest_component(mask_bright)
    fg_dark = largest_component(mask_dark)

    # Choose the one with larger area
    fg = fg_bright if fg_bright.sum() >= fg_dark.sum() else fg_dark

    if fg.sum() == 0:
        return 0, None, None, None, None

    props = regionprops(fg)
    (cy, cx) = props[0].centroid  # regionprops gives (row, col)

    d = float(np.hypot(cx - x0, cy - y0))
    R = min(W, H) / 2.0
    r = min(1.0, d / R)

    Q2 = int(round((1 - r) * 100))
    Q2 = max(0, min(100, Q2))

    return Q2, cx, cy, d, r
