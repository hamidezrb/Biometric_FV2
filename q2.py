
# Offset Complement
"""The offset complement of a vascular image is a centering score that quantifies
how close the foreground region centroid is to the image geometric center. 
It is computed from normalized horizontal and vertical centroid offsets on the image plane."""

import cv2
import numpy as np
from typing import Tuple, Optional

# def calculate_q2(image_path: str) -> Tuple[int, Optional[float], Optional[float], Optional[float], Optional[float]]:
#     """
#     ISO/IEC 29794-9 Quality Component Q2 – Offset Complement (Centering)
#     """

#     img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     if img is None:
#         print(f"[Q2] ERROR: Cannot read {image_path}")
#         return 0, None, None, None, None

#     H, W = img.shape
#     gx, gy = W / 2.0, H / 2.0  # geometric center

#     # --- Foreground extraction (Otsu thresholding)
#     _, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     mask_inv = cv2.bitwise_not(mask)

#     # Pick whichever has larger foreground area (bright or dark)
#     if np.sum(mask) < np.sum(mask_inv):
#         mask = mask_inv

#     # Find largest connected component
#     num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
#     if num_labels <= 1:
#         return 0, None, None, None, None
#     areas = stats[1:, cv2.CC_STAT_AREA]
#     largest_label = np.argmax(areas) + 1
#     fg = (labels == largest_label).astype(np.uint8) * 255

#     # --- Compute centroid of foreground
#     m = cv2.moments(fg)
#     # total area = sum of pixel values / 255
#     if m['m00'] == 0:
#         return 0, None, None, None, None
#     # m['m10'] = Σ (x * pixel_value)
#     # m['m01'] = Σ (y * pixel_value)
#     cx = m['m10'] / m['m00']
#     cy = m['m01'] / m['m00']

#     # --- Compute normalized offsets (ISO formula)
#     S_H = abs(cx - gx) / gx
#     S_V = abs(cy - gy) / gy
#     r = np.sqrt(S_H**2 + S_V**2)

#     # --- Compute final Q2
#     Q2 = int(round((1 - r) * 100))
#     Q2 = max(0, min(100, Q2))

#     # will be closer to the right side of the image → S_H increases → Q₂ decreases.
#     # If finger is centered: Cₓ moves to center → Q₂ increases.
#     return Q2, cx, cy, S_H, S_V




# ******************************ISO-aligned*************************************
 

def calculate_q2(
    R_mask: np.ndarray,
    grayscale_image: np.ndarray
) -> Tuple[int, Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    ISO/IEC 29794-9 Quality Component Q2 – Offset Complement (Centering)

    ISO intent:
    - Use the foreground region R (same as Q1).
    - Compute centroid of R, compare to image geometric center.
    - Q2 = round((1 - r) * 100), clipped to [0,100]
      where r = sqrt(S_H^2 + S_V^2),
            S_H = |cx - gx| / gx,
            S_V = |cy - gy| / gy

    Args:
        R_mask: uint8 mask, 255 = foreground (R), 0 = background
        grayscale_image: grayscale image (used only for shape)

    Returns:
        (Q2, cx, cy, S_H, S_V)
    """

    if R_mask is None or grayscale_image is None:
        return 0, None, None, None, None

    if R_mask.shape != grayscale_image.shape:
        raise ValueError("R_mask and grayscale_image must have the same shape")

    H, W = grayscale_image.shape
    gx, gy = W / 2.0, H / 2.0  # geometric center

    # Foreground pixels
    fg = (R_mask == 255)
    if np.count_nonzero(fg) == 0:
        # Veto: invalid R
        return 0, None, None, None, None

    # Compute centroid (cx, cy) of foreground region R
    # Using image moments on a 0/1 mask is stable.
    fg_u8 = fg.astype(np.uint8)
    m = cv2.moments(fg_u8, binaryImage=True)
    if m["m00"] == 0:
        return 0, None, None, None, None

    cx = m["m10"] / m["m00"]
    cy = m["m01"] / m["m00"]

    # Normalized offsets
    # Avoid division by zero if gx or gy are 0 (degenerate images)
    if gx == 0 or gy == 0:
        return 0, cx, cy, None, None

    S_H = abs(cx - gx) / gx
    S_V = abs(cy - gy) / gy
    r = float(np.sqrt(S_H**2 + S_V**2))

    Q2 = int(round((1.0 - r) * 100.0))
    Q2 = max(0, min(100, Q2))

    return Q2, float(cx), float(cy), float(S_H), float(S_V)


