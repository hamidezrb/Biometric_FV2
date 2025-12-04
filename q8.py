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


def calculate_q8(R_mask: np.ndarray,
                 Grayscale_Image: np.ndarray,
                 Lc: int = 600) -> Tuple[int, int]:
    """
    Calculate Q8 (Total Vascular Length) following ISO/IEC 29794-9 Clause 5.2.8.
    
    Args:
        R_mask (np.ndarray): Binary mask of foreground region (255 for foreground, 0 for background)
        Grayscale_Image (np.ndarray): Original grayscale image
        Lc (int): Coefficient for vascular length normalization (default: 600 for Finger, Second phalanx)
        
    Returns:
        Tuple[int, int]:
            (Q8_score, N_vessel)
            - Q8_score: Final integer Q8 score (0-100)
            - N_vessel: Total number of pixels in thinned vessel skeleton
    
    ISO Requirements [Clause 5.2.8]:
    1. Veto Logic: If R_mask is invalid, return Q8 = 0
    2. Vessel Extraction Pipeline:
       a. Binarize the Grayscale_Image using the R_mask
       b. Skeletonize (thin) the binarized veins to 1-pixel thickness
    3. Measurement: Count total pixels in thinned image (N_vessel)
    4. Q8 Calculation (Formula 15): Q8 = MIN(100, ROUND(N_vessel / Lc * 100))
    """
    # Veto Logic: If R_mask is invalid (area=0), return Q8 = 0
    foreground_mask = (R_mask == 255)
    if np.count_nonzero(foreground_mask) == 0:
        return 0, 0
    
    # Step 1: Binarize the grayscale image within the R_mask region
    # Use Otsu thresholding on the foreground region only
    gray = Grayscale_Image.astype(np.uint8)
    
    # Extract foreground region for thresholding
    foreground_region = gray[foreground_mask]
    if foreground_region.size == 0:
        return 0, 0
    
    # Apply Otsu thresholding to the foreground region
    threshold_value, _ = cv2.threshold(foreground_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Create binarized image: apply threshold within R_mask, set background to 0
    binarized = np.zeros_like(gray, dtype=np.uint8)
    binarized[foreground_mask] = (gray[foreground_mask] > threshold_value).astype(np.uint8) * 255
    
    # Step 2: Skeletonize (thin) the binarized image to 1-pixel thickness
    skeleton = _skeletonize(binarized)
    
    # Step 3: Count total pixels in thinned skeleton (N_vessel)
    # Only count pixels within the R_mask region
    vessel_pixels = (skeleton == 255) & foreground_mask
    N_vessel = int(np.count_nonzero(vessel_pixels))
    
    # Step 4: Calculate Q8 using Formula 15
    # Q8 = MIN(100, ROUND(N_vessel / Lc * 100))
    if Lc <= 0:
        return 0, N_vessel
    
    q8_raw = (N_vessel / float(Lc)) * 100.0
    Q8_score = int(round(q8_raw))
    Q8_score = max(0, min(100, Q8_score))
    
    return Q8_score, N_vessel

