"""
ISO/IEC 29794-9 Quality Component 8 (Total Vascular Length) Implementation
Finger Vascular Biometrics - Second Phalanx

This module implements the mandatory ISO calculation for Q8 (Total Vascular Length)
according to ISO/IEC 29794-9 Clause 5.2.8.

Requirements:
- Vessel Extraction Pipeline: Binarization + Skeletonization (Thinning)
- Measurement: Count total pixels in thinned vessel skeleton (N_vessel)
- Q8 Calculation (Formula 15): Q8 = MIN(100, ROUND(N_vessel / Lc * 100))
- Lc = 600 pixels (for Finger, Second phalanx)
"""

import cv2
import numpy as np
from typing import Tuple


def _skeletonize(binarized_image: np.ndarray) -> np.ndarray:
    """
    Skeletonize a binary image using iterative morphological thinning (Zhang-Suen algorithm).
    Reduces vessel structures to 1-pixel thickness.
    
    Args:
        binarized_image (np.ndarray): Binary image (0 = background, 255 = foreground)
        
    Returns:
        np.ndarray: Skeletonized binary image (0 = background, 255 = foreground)
    """
    # Ensure binary image (0 or 255)
    if binarized_image.dtype != np.uint8:
        binarized_image = binarized_image.astype(np.uint8)
    
    # Convert to binary (0 or 1) for thinning algorithm
    binary = (binarized_image > 127).astype(np.uint8)
    
    # Zhang-Suen thinning algorithm
    # Create structuring elements for iterative thinning
    # This is a standard morphological thinning approach
    skeleton = binary.copy()
    prev = np.zeros_like(skeleton)
    
    # Iterate until no more changes
    max_iterations = 1000
    iteration = 0
    
    while iteration < max_iterations:
        prev[:] = skeleton[:]
        
        # Sub-iteration 1: Mark pixels for deletion
        markers = np.zeros_like(skeleton)
        for i in range(1, skeleton.shape[0] - 1):
            for j in range(1, skeleton.shape[1] - 1):
                if skeleton[i, j] == 0:
                    continue
                
                # Get 8-neighborhood
                p2 = skeleton[i-1, j]
                p3 = skeleton[i-1, j+1]
                p4 = skeleton[i, j+1]
                p5 = skeleton[i+1, j+1]
                p6 = skeleton[i+1, j]
                p7 = skeleton[i+1, j-1]
                p8 = skeleton[i, j-1]
                p9 = skeleton[i-1, j-1]
                
                # Count transitions from 0 to 1
                transitions = sum([
                    (p2 == 0 and p3 == 1),
                    (p3 == 0 and p4 == 1),
                    (p4 == 0 and p5 == 1),
                    (p5 == 0 and p6 == 1),
                    (p6 == 0 and p7 == 1),
                    (p7 == 0 and p8 == 1),
                    (p8 == 0 and p9 == 1),
                    (p9 == 0 and p2 == 1)
                ])
                
                # Count neighbors
                neighbors = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
                
                # Zhang-Suen conditions for sub-iteration 1
                if (2 <= neighbors <= 6 and
                    transitions == 1 and
                    p2 * p4 * p6 == 0 and
                    p4 * p6 * p8 == 0):
                    markers[i, j] = 1
        
        skeleton = skeleton & (~markers)
        
        # Sub-iteration 2: Mark pixels for deletion
        markers[:] = 0
        for i in range(1, skeleton.shape[0] - 1):
            for j in range(1, skeleton.shape[1] - 1):
                if skeleton[i, j] == 0:
                    continue
                
                # Get 8-neighborhood
                p2 = skeleton[i-1, j]
                p3 = skeleton[i-1, j+1]
                p4 = skeleton[i, j+1]
                p5 = skeleton[i+1, j+1]
                p6 = skeleton[i+1, j]
                p7 = skeleton[i+1, j-1]
                p8 = skeleton[i, j-1]
                p9 = skeleton[i-1, j-1]
                
                # Count transitions from 0 to 1c
                transitions = sum([
                    (p2 == 0 and p3 == 1),
                    (p3 == 0 and p4 == 1),
                    (p4 == 0 and p5 == 1),
                    (p5 == 0 and p6 == 1),
                    (p6 == 0 and p7 == 1),
                    (p7 == 0 and p8 == 1),
                    (p8 == 0 and p9 == 1),
                    (p9 == 0 and p2 == 1)
                ])
                
                # Count neighbors
                neighbors = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
                
                # Zhang-Suen conditions for sub-iteration 2
                if (2 <= neighbors <= 6 and
                    transitions == 1 and
                    p2 * p4 * p8 == 0 and
                    p2 * p6 * p8 == 0):
                    markers[i, j] = 1
        
        skeleton = skeleton & (~markers)
        
        # Check if converged
        if np.array_equal(skeleton, prev):
            break
        
        iteration += 1
    
    # Convert back to 0-255 format
    skeleton_255 = skeleton.astype(np.uint8) * 255
    return skeleton_255


# def calculate_q8(R_mask: np.ndarray,
#                  Grayscale_Image: np.ndarray,
#                  Lc: int = 600) -> Tuple[int, int]:
#     """
#     Calculate Q8 (Total Vascular Length) following ISO/IEC 29794-9 Clause 5.2.8.
    
#     Args:
#         R_mask (np.ndarray): Binary mask of foreground region (255 for foreground, 0 for background)
#         Grayscale_Image (np.ndarray): Original grayscale image
#         Lc (int): Coefficient for vascular length normalization (default: 600 for Finger, Second phalanx)
        
#     Returns:
#         Tuple[int, int]:
#             (Q8_score, N_vessel)
#             - Q8_score: Final integer Q8 score (0-100)
#             - N_vessel: Total number of pixels in thinned vessel skeleton
    
#     ISO Requirements [Clause 5.2.8]:
#     1. Veto Logic: If R_mask is invalid, return Q8 = 0
#     2. Vessel Extraction Pipeline:
#        a. Binarize the Grayscale_Image using the R_mask
#        b. Skeletonize (thin) the binarized veins to 1-pixel thickness
#     3. Measurement: Count total pixels in thinned image (N_vessel)
#     4. Q8 Calculation (Formula 15): Q8 = MIN(100, ROUND(N_vessel / Lc * 100))
#     """
#     # Veto Logic: If R_mask is invalid (area=0), return Q8 = 0
#     foreground_mask = (R_mask == 255)
#     if np.count_nonzero(foreground_mask) == 0:
#         return 0, 0
    
#     # Step 1: Binarize the grayscale image within the R_mask region
#     # Use Otsu thresholding on the foreground region only
#     gray = Grayscale_Image.astype(np.uint8)
    
#     # Extract foreground region for thresholding
#     foreground_region = gray[foreground_mask]
#     if foreground_region.size == 0:
#         return 0, 0
    
#     # Apply Otsu thresholding to the foreground region
#     threshold_value, _ = cv2.threshold(foreground_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
#     # Create binarized image: apply threshold within R_mask, set background to 0
#     binarized = np.zeros_like(gray, dtype=np.uint8)
#     binarized[foreground_mask] = (gray[foreground_mask] > threshold_value).astype(np.uint8) * 255
    
#     # Step 2: Skeletonize (thin) the binarized image to 1-pixel thickness
#     skeleton = _skeletonize(binarized)
    
#     # Step 3: Count total pixels in thinned skeleton (N_vessel)
#     # Only count pixels within the R_mask region
#     vessel_pixels = (skeleton == 255) & foreground_mask
#     N_vessel = int(np.count_nonzero(vessel_pixels))
    
#     # Step 4: Calculate Q8 using Formula 15
#     # Q8 = MIN(100, ROUND(N_vessel / Lc * 100))
#     if Lc <= 0:
#         return 0, N_vessel
    
#     q8_raw = (N_vessel / float(Lc)) * 100.0
#     Q8_score = int(round(q8_raw))
#     Q8_score = max(0, min(100, Q8_score))
    
#     return Q8_score, N_vessel




# ******************************ISO-aligned*************************************

# def _skeletonize_opencv_or_zhang(binary255: np.ndarray) -> np.ndarray:
#     """
#     Skeletonize a binary image to 1-pixel width.
#     Prefers OpenCV ximgproc thinning if available, otherwise uses a Zhang–Suen fallback.
#     Returns uint8 image 0/255.
#     """
#     # Prefer OpenCV contrib thinning if available
#     try:
#         b01 = (binary255 > 0).astype(np.uint8)
#         sk = cv2.ximgproc.thinning(b01 * 255, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
#         return (sk > 0).astype(np.uint8) * 255
#     except Exception:
#         pass

#     # Fallback: simple Zhang–Suen thinning (your previous implementation idea)
#     binary = (binary255 > 0).astype(np.uint8)
#     skeleton = binary.copy()
#     prev = np.zeros_like(skeleton)

#     for _ in range(1000):
#         prev[:] = skeleton[:]

#         # sub-iter 1
#         markers = np.zeros_like(skeleton)
#         for i in range(1, skeleton.shape[0] - 1):
#             for j in range(1, skeleton.shape[1] - 1):
#                 if skeleton[i, j] == 0:
#                     continue
#                 p2 = skeleton[i-1, j]
#                 p3 = skeleton[i-1, j+1]
#                 p4 = skeleton[i, j+1]
#                 p5 = skeleton[i+1, j+1]
#                 p6 = skeleton[i+1, j]
#                 p7 = skeleton[i+1, j-1]
#                 p8 = skeleton[i, j-1]
#                 p9 = skeleton[i-1, j-1]

#                 transitions = sum([
#                     (p2 == 0 and p3 == 1),
#                     (p3 == 0 and p4 == 1),
#                     (p4 == 0 and p5 == 1),
#                     (p5 == 0 and p6 == 1),
#                     (p6 == 0 and p7 == 1),
#                     (p7 == 0 and p8 == 1),
#                     (p8 == 0 and p9 == 1),
#                     (p9 == 0 and p2 == 1),
#                 ])
#                 neighbors = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9

#                 if (2 <= neighbors <= 6 and transitions == 1 and
#                     p2 * p4 * p6 == 0 and p4 * p6 * p8 == 0):
#                     markers[i, j] = 1

#         skeleton = skeleton & (1 - markers)

#         # sub-iter 2
#         markers[:] = 0
#         for i in range(1, skeleton.shape[0] - 1):
#             for j in range(1, skeleton.shape[1] - 1):
#                 if skeleton[i, j] == 0:
#                     continue
#                 p2 = skeleton[i-1, j]
#                 p3 = skeleton[i-1, j+1]
#                 p4 = skeleton[i, j+1]
#                 p5 = skeleton[i+1, j+1]
#                 p6 = skeleton[i+1, j]
#                 p7 = skeleton[i+1, j-1]
#                 p8 = skeleton[i, j-1]
#                 p9 = skeleton[i-1, j-1]

#                 transitions = sum([
#                     (p2 == 0 and p3 == 1),
#                     (p3 == 0 and p4 == 1),
#                     (p4 == 0 and p5 == 1),
#                     (p5 == 0 and p6 == 1),
#                     (p6 == 0 and p7 == 1),
#                     (p7 == 0 and p8 == 1),
#                     (p8 == 0 and p9 == 1),
#                     (p9 == 0 and p2 == 1),
#                 ])
#                 neighbors = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9

#                 if (2 <= neighbors <= 6 and transitions == 1 and
#                     p2 * p4 * p8 == 0 and p2 * p6 * p8 == 0):
#                     markers[i, j] = 1

#         skeleton = skeleton & (1 - markers)

#         if np.array_equal(skeleton, prev):
#             break

#     return skeleton.astype(np.uint8) * 255


# def _extract_vessels(gray: np.ndarray, fg_mask: np.ndarray) -> np.ndarray:
#     """
#     Robust vessel extraction (ISO/PWI compatible: binarize vascular pattern inside R).

#     Steps:
#     - CLAHE (contrast)
#     - Black-hat (enhance dark lines)
#     - Threshold on foreground-only pixels:
#         * try Otsu on non-zero foreground values
#         * sanity-check sparsity/density
#         * fallback to percentile threshold if Otsu collapses
#     - Small cleanup (open)
#     """
#     roi = gray.copy()
#     roi[~fg_mask] = 0

#     # Contrast enhancement
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     roi_c = clahe.apply(roi)

#     # Slight blur to reduce speckle before morphology
#     roi_c = cv2.GaussianBlur(roi_c, (3, 3), 0)

#     # Kernel size relative to image size (important!)
#     H, W = gray.shape
#     k = int(round(min(H, W) * 0.03))
#     k = max(9, k)           # minimum
#     if k % 2 == 0:
#         k += 1              # odd size
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

#     # Black-hat: dark lines -> bright response
#     bh = cv2.morphologyEx(roi_c, cv2.MORPH_BLACKHAT, kernel)

#     # Use only foreground non-zero values for thresholding
#     vals = bh[fg_mask]
#     vals = vals[vals > 0]

#     if vals.size < 50:
#         # Not enough information -> no vessels
#         vessel = np.zeros_like(gray, dtype=np.uint8)
#         vessel[~fg_mask] = 0
#         return vessel

#     # --- Try Otsu on foreground values only (reshape required by OpenCV)
#     vals_reshaped = vals.reshape(-1, 1)
#     t, _ = cv2.threshold(vals_reshaped, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     t = float(t)

#     # Apply threshold to bh image
#     vessel = (bh >= t).astype(np.uint8) * 255
#     vessel[~fg_mask] = 0

#     # --- Sanity check: if too sparse/dense, fallback to percentile threshold
#     fg_count = int(np.count_nonzero(fg_mask))
#     vessel_count = int(np.count_nonzero(vessel))

#     # These ratios are dataset-tunable but generally safe:
#     sparse_ratio = 0.002   # <0.2% of ROI pixels = too sparse
#     dense_ratio  = 0.12    # >12% of ROI pixels = too dense

#     ratio = vessel_count / max(1, fg_count)

#     if ratio < sparse_ratio or ratio > dense_ratio:
#         # Percentile fallback: keep strongest responses only
#         # 90-97 works well; choose 95 as default
#         p = 95
#         tp = float(np.percentile(vals, p))
#         vessel = (bh >= tp).astype(np.uint8) * 255
#         vessel[~fg_mask] = 0

#     # Cleanup (remove isolated dots)
#     vessel = cv2.morphologyEx(
#         vessel, cv2.MORPH_OPEN,
#         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
#     )
#     vessel[~fg_mask] = 0
#     return vessel



# def calculate_q8(
#     R_mask: np.ndarray,
#     Grayscale_Image: np.ndarray,
#     Lc: int = 600,
#     return_debug: bool = False
# ) -> Tuple[int, int] | Tuple[int, int, np.ndarray, np.ndarray]:
#     """
#     Q8 — Total Vascular Length (ISO/IEC 29794-9 PWI draft, Clause 5.2.8 style)

#     Required steps:
#     1) Veto if R invalid
#     2) Binarize vascular pattern inside R
#     3) Thin (skeletonize)
#     4) Count skeleton pixels -> N_vessel
#     5) Q8 = min(100, round(N_vessel / Lc * 100))

#     Args:
#         R_mask: uint8 mask (255=foreground R)
#         Grayscale_Image: grayscale image
#         Lc: 600 for finger, 2nd phalanx
#         return_debug: if True, also returns (vessel_binary, skeleton)

#     Returns:
#         (Q8_score, N_vessel) or (Q8_score, N_vessel, vessel_binary, skeleton)
#     """
#     fg = (R_mask == 255)
#     if np.count_nonzero(fg) == 0:
#         if return_debug:
#             z = np.zeros_like(Grayscale_Image, dtype=np.uint8)
#             return 0, 0, z, z
#         return 0, 0

#     gray = Grayscale_Image.astype(np.uint8)

#     # 1) Extract vessel binary (vessels=255)
#     vessel_binary = _extract_vessels(gray, fg)

#     # 2) Skeletonize (thin)
#     skeleton = _skeletonize_opencv_or_zhang(vessel_binary)

#     # 3) Count vessel pixels inside R
#     N_vessel = int(np.count_nonzero((skeleton == 255) & fg))

#     # 4) Score
#     if Lc <= 0:
#         Q8 = 0
#     else:
#         Q8 = int(round((N_vessel / float(Lc)) * 100.0))
#         Q8 = max(0, min(100, Q8))

#     if return_debug:
#         return Q8, N_vessel, vessel_binary, skeleton
#     return Q8, N_vessel

"""
1.Extract foreground region R
2.Binarize vascular pattern in R
3.Thin (skeletonize) to 1-pixel thickness
4.Count vessel skeleton pixels N_vessel 
5.Map to [0,100] , Calculate Q8 = min(100, round(N_vessel / Lc * 100))

"""

import cv2
import numpy as np
from typing import Tuple, Union


def remove_small_components(binary255: np.ndarray, min_area: int = 30) -> np.ndarray:
    """
    Remove connected components smaller than min_area.
    Input/output: uint8 0/255.
    """
    b01 = (binary255 > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(b01, connectivity=8)
    out = np.zeros_like(binary255, dtype=np.uint8)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 255
    return out


def skeletonize_binary(binary255: np.ndarray) -> np.ndarray:
    """
    Skeletonize (thin) to 1-pixel width.
    Input: uint8 0/255
    Output: uint8 0/255
    """
    b01 = (binary255 > 0).astype(np.uint8)

    # Prefer OpenCV contrib thinning if available
    try:
        sk = cv2.ximgproc.thinning(b01 * 255, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
        return (sk > 0).astype(np.uint8) * 255
    except Exception:
        pass

    # Fallback: morphological skeleton (works without contrib)
    img = b01.copy() * 255
    skel = np.zeros_like(img, dtype=np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while True:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded
        if cv2.countNonZero(img) == 0:
            break

    return (skel > 0).astype(np.uint8) * 255


def extract_vessels_f_ex_style(
    gray: np.ndarray,
    fg_mask: np.ndarray,
    block_size: int = 31,
    C: int = 5,
    median_ksize: int = 5,
    close_iters: int = 1,
    min_area: int = 30
) -> np.ndarray:
    """
    Vessel extraction inspired by f_ex.py, but aligned for ISO Q8/Q9 use:
    - Works inside foreground region R (fg_mask)
    - No flip/crop
    - Output is binary vessels=255, background=0
    """
    # Mask ROI
    roi = gray.copy()
    roi[~fg_mask] = 0

    # Mild contrast enhancement helps adaptive threshold stability
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    roi = clahe.apply(roi)

    # Blur like f_ex.py (median blur)
    if median_ksize % 2 == 0:
        median_ksize += 1
    roi_blur = cv2.medianBlur(roi, median_ksize)

    # Adaptive threshold (Gaussian) like f_ex.py
    # Ensure odd block size >= 3
    if block_size % 2 == 0:
        block_size += 1
    block_size = max(3, block_size)

    thr = cv2.adaptiveThreshold(
        roi_blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        C
    )

    # adaptiveThreshold produces bright background often; we want vessels=white
    # So invert and then apply fg mask.
    vessel = cv2.bitwise_not(thr)
    vessel[~fg_mask] = 0

    # Bridge small gaps (reduces fragmentation -> stabilizes Q8/Q9)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    if close_iters > 0:
        vessel = cv2.morphologyEx(vessel, cv2.MORPH_CLOSE, se, iterations=close_iters)

    # Remove speckles
    vessel = cv2.morphologyEx(vessel, cv2.MORPH_OPEN, se, iterations=1)

    # Remove tiny components (dots)
    if min_area > 1:
        vessel = remove_small_components(vessel, min_area=min_area)

    vessel[~fg_mask] = 0
    return vessel


def calculate_q8(
    R_mask: np.ndarray,
    Grayscale_Image: np.ndarray,
    Lc: int = 600,
    return_debug: bool = False,
    # you can tune these slightly if needed
    block_size: int = 31,
    C: int = 5,
    median_ksize: int = 5,
    close_iters: int = 1,
    min_area: int = 30
) -> Union[Tuple[int, int], Tuple[int, int, np.ndarray, np.ndarray]]:
    """
    Q8 — Total Vascular Length (ISO/IEC 29794-9 PWI draft style)

    ISO/PWI intent:
    1) Use foreground region R (R_mask)
    2) Binarize vascular pattern in R
    3) Thin (skeletonize) to 1-pixel thickness
    4) Count vessel skeleton pixels N_vessel
    5) Q8 = min(100, round(N_vessel/Lc * 100))

    Returns:
        (Q8_score, N_vessel) or (Q8_score, N_vessel, vessel_binary, skeleton)
    """
    if R_mask is None or Grayscale_Image is None:
        if return_debug:
            z = np.zeros((1, 1), dtype=np.uint8)
            return 0, 0, z, z
        return 0, 0

    if R_mask.shape != Grayscale_Image.shape:
        raise ValueError("R_mask and Grayscale_Image must have same shape")

    fg = (R_mask == 255)
    if np.count_nonzero(fg) == 0:
        if return_debug:
            z = np.zeros_like(Grayscale_Image, dtype=np.uint8)
            return 0, 0, z, z
        return 0, 0

    gray = Grayscale_Image.astype(np.uint8)

    # 1) Vessel extraction (f_ex style, but mask-consistent)
    vessel_binary = extract_vessels_f_ex_style(
        gray, fg,
        block_size=block_size,
        C=C,
        median_ksize=median_ksize,
        close_iters=close_iters,
        min_area=min_area
    )

    # 2) Thin to skeleton
    skeleton = skeletonize_binary(vessel_binary)
    skeleton[~fg] = 0

    # 3) Count skeleton pixels
    N_vessel = int(np.count_nonzero(skeleton == 255))

    # 4) Score
    if Lc <= 0:
        Q8 = 0
    else:
        Q8 = int(round((N_vessel / float(Lc)) * 100.0))
        Q8 = max(0, min(100, Q8))

    if return_debug:
        return Q8, N_vessel, vessel_binary, skeleton
    return Q8, N_vessel
