import cv2
import numpy as np
from typing import Tuple, Optional


def calculate_q2(image_path: str) -> Tuple[int, Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    Q2 — Offset Complement (Centering) using OpenCV only.
    Steps:
      1. Read image as grayscale.
      2. Otsu threshold (both dark and bright masks).
      3. Extract largest connected component.
      4. Compute centroid (cx, cy) using moments.
      5. Measure distance from image center.
      6. Normalize and map to [0,100].
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"[Q2] ERROR: Cannot read {image_path}")
        return 0, None, None, None, None

    H, W = img.shape
    x0, y0 = (W - 1) / 2.0, (H - 1) / 2.0

    # Otsu thresholding using OpenCV (same as Q1)
    try:
        T, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    except Exception:
        T = np.median(img)

    mask_bright = (img >= T).astype(np.uint8) * 255
    mask_dark = (img <= T).astype(np.uint8) * 255

    def largest_component(mask):
        """Find largest connected component using OpenCV."""
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        if num_labels <= 1:  # Only background (label 0)
            return np.zeros_like(mask)
        
        # Find largest component (skip label 0 which is background)
        # stats format: [x, y, width, height, area]
        areas = stats[1:, cv2.CC_STAT_AREA]  # Get areas for all components except background
        largest_label = np.argmax(areas) + 1  # +1 because we skipped background (label 0)
        
        # Create mask with only the largest component
        component_mask = (labels == largest_label).astype(np.uint8) * 255
        return component_mask

    fg_bright = largest_component(mask_bright)
    fg_dark = largest_component(mask_dark)

    # Choose the one with larger area
    fg = fg_bright if fg_bright.sum() >= fg_dark.sum() else fg_dark

    if fg.sum() == 0:
        return 0, None, None, None, None

    # Compute centroid using moments (OpenCV method)
    moments = cv2.moments(fg)
    if moments['m00'] == 0:
        return 0, None, None, None, None
    
    cx = moments['m10'] / moments['m00']  # x-coordinate (column)
    cy = moments['m01'] / moments['m00']  # y-coordinate (row)

    d = float(np.hypot(cx - x0, cy - y0))
    R = min(W, H) / 2.0
    r = min(1.0, d / R)

    Q2 = int(round((1 - r) * 100))
    Q2 = max(0, min(100, Q2))

    return Q2, cx, cy, d, r
