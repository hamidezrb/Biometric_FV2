
# Offset Complement
"""The offset complement of a vascular image is a centering score that quantifies
how close the foreground region centroid is to the image geometric center. 
It is computed from normalized horizontal and vertical centroid offsets on the image plane."""

import cv2
import numpy as np
from typing import Tuple, Optional

def calculate_q2(image_path: str) -> Tuple[int, Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    ISO/IEC 29794-9 Quality Component Q2 – Offset Complement (Centering)
    """

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"[Q2] ERROR: Cannot read {image_path}")
        return 0, None, None, None, None

    H, W = img.shape
    gx, gy = W / 2.0, H / 2.0  # geometric center

    # --- Foreground extraction (Otsu thresholding)
    _, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask_inv = cv2.bitwise_not(mask)

    # Pick whichever has larger foreground area (bright or dark)
    if np.sum(mask) < np.sum(mask_inv):
        mask = mask_inv

    # Find largest connected component
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return 0, None, None, None, None
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = np.argmax(areas) + 1
    fg = (labels == largest_label).astype(np.uint8) * 255

    # --- Compute centroid of foreground
    m = cv2.moments(fg)
    # total area = sum of pixel values / 255
    if m['m00'] == 0:
        return 0, None, None, None, None
    # m['m10'] = Σ (x * pixel_value)
    # m['m01'] = Σ (y * pixel_value)
    cx = m['m10'] / m['m00']
    cy = m['m01'] / m['m00']

    # --- Compute normalized offsets (ISO formula)
    S_H = abs(cx - gx) / gx
    S_V = abs(cy - gy) / gy
    r = np.sqrt(S_H**2 + S_V**2)

    # --- Compute final Q2
    Q2 = int(round((1 - r) * 100))
    Q2 = max(0, min(100, Q2))

    # will be closer to the right side of the image → S_H increases → Q₂ decreases.
    # If finger is centered: Cₓ moves to center → Q₂ increases.
    return Q2, cx, cy, S_H, S_V
