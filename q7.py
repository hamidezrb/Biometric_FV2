"""
ISO/IEC 29794-9 Quality Component 7 (Brightness Uniformity) Implementation
Finger Vascular Biometrics - Second Phalanx

This module implements the mandatory ISO calculation for Q7 (Brightness Uniformity)
according to ISO/IEC 29794-9 Clause 5.2.7.

Requirements:
- Input: R_mask (Foreground Mask), Grayscale_Image, and overall Foreground Mean (g_mean, which is x_mean)
- Segmentation: Divide the image into 5x5 pixel blocks (overlapping)
- Local Mean Calculation: For each block (Nb), calculate the local mean (xi_mean) only using pixels inside R_mask (Formula 13)
- Q7 Calculation using Formula (14): Q7 = ROUND( 1 - (1/121) * SQRT( (1/Nb) * SUM( (xi_mean - x_mean)^2 ) ) * 100 )
- Q7 must be integer (0-100)
"""

import numpy as np
from typing import Tuple


def calculate_q7(R_mask: np.ndarray, Grayscale_Image: np.ndarray, g_mean: float) -> Tuple[int, float]:
    """
    Calculate Q7 (Brightness Uniformity) quality metric.
    
    Args:
        R_mask (np.ndarray): Binary mask of foreground region (255 for foreground, 0 for background)
        Grayscale_Image (np.ndarray): Original grayscale image
        g_mean (float): Overall foreground mean (x_mean) - typically from Q3 calculation
        
    Returns:
        Tuple[int, float]: (Q7_score, block_variance)
            - Q7_score: Final integer Q7 score (0-100)
            - block_variance: Variance of block means for debugging
    
    ISO Requirements [Clause 5.2.7]:
    1. Veto Logic: If R_mask is invalid (area=0), return Q7 = 0
    2. Segmentation: Divide image into 5x5 pixel blocks (overlapping)
    3. Local Mean Calculation: For each block, calculate xi_mean using only pixels inside R_mask (Formula 13)
    4. Q7 Calculation using Formula (14): Q7 = ROUND( 1 - (1/121) * SQRT( (1/Nb) * SUM( (xi_mean - x_mean)^2 ) ) * 100 )
    5. Q7 must be integer (0-100)
    """
    
    # Veto Logic: If R_mask is invalid (area=0), return Q7 = 0
    foreground_mask = (R_mask == 255)
    N = np.sum(foreground_mask)
    
    if N == 0:
        return 0, 0.0
    
    # Get image dimensions
    H, W = Grayscale_Image.shape
    
    # Block size is 5x5
    block_size = 5
    
    # Collect local means (xi_mean) for each 5x5 block that overlaps with foreground
    # We use overlapping blocks (sliding window approach)
    block_means = []
    
    # Slide 5x5 window across the image
    for y in range(H - block_size + 1):
        for x in range(W - block_size + 1):
            # Extract 5x5 block region
            block_mask = R_mask[y:y+block_size, x:x+block_size]
            block_image = Grayscale_Image[y:y+block_size, x:x+block_size]
            
            # Only consider blocks that have at least some foreground pixels
            block_foreground = (block_mask == 255)
            block_foreground_count = np.sum(block_foreground)
            
            if block_foreground_count > 0:
                # Calculate local mean (xi_mean) using only pixels inside R_mask (Formula 13)
                block_foreground_pixels = block_image[block_foreground]
                xi_mean = np.mean(block_foreground_pixels)
                block_means.append(xi_mean)
    
    # Nb = number of blocks with foreground pixels
    Nb = len(block_means)
    
    if Nb == 0:
        return 0, 0.0
    
    # x_mean is the overall foreground mean (g_mean passed as parameter)
    x_mean = g_mean
    
    # Calculate variance of block means: (1/Nb) * SUM( (xi_mean - x_mean)^2 )
    block_means_array = np.array(block_means)
    squared_diff = (block_means_array - x_mean) ** 2
    block_variance = np.sum(squared_diff) / Nb
    
    # Q7 Calculation using Formula (14):
    # Q7 = ROUND( 1 - (1/121) * SQRT( (1/Nb) * SUM( (xi_mean - x_mean)^2 ) ) * 100 )
    # Note: The factor 1/121 is a normalization constant (121 = 11^2, related to block size normalization)
    sqrt_variance = np.sqrt(block_variance)
    q7_raw = (1 - (1.0 / 121.0) * sqrt_variance) * 100
    Q7_score = round(q7_raw)
    
    # Cap at [0, 100]
    Q7_score = max(0, min(100, Q7_score))
    
    return Q7_score, block_variance




# ******************************ISO-aligned*************************************


import numpy as np
from typing import Tuple


def calculate_q7_ISO(
    R_mask: np.ndarray,
    Grayscale_Image: np.ndarray,
    g_mean: float,
    block_size: int = 5,
    min_fg_ratio: float = 0.5
) -> Tuple[int, float]:
    """
    Q7 (Brightness Uniformity) — ISO/IEC 29794-9 Clause 5.2.7

    Fix:
    - Only include blocks with sufficient foreground coverage.
      This avoids unstable means from blocks that intersect the ROI boundary.

    Args:
        R_mask: uint8 mask, 255=foreground, 0=background
        Grayscale_Image: grayscale image
        g_mean: global mean inside foreground region (x_mean)
        block_size: block size (default 5)
        min_fg_ratio: minimum ratio of pixels inside R required to accept a block (default 0.5)

    Returns:
        (Q7_score, block_variance)
    """
    fg = (R_mask == 255)
    if np.count_nonzero(fg) == 0:
        return 0, 0.0

    H, W = Grayscale_Image.shape
    bs = int(block_size)
    if bs <= 0 or bs > H or bs > W:
        return 0, 0.0

    # Minimum number of foreground pixels required in a block
    min_fg_count = int(np.ceil(min_fg_ratio * (bs * bs)))

    block_means = []

    for y in range(H - bs + 1):
        for x in range(W - bs + 1):
            block_mask = fg[y:y+bs, x:x+bs]
            fg_count = int(np.count_nonzero(block_mask))

            # ISO-robust: ignore blocks with too little ROI support
            if fg_count < min_fg_count:
                continue

            block_img = Grayscale_Image[y:y+bs, x:x+bs]
            xi_mean = float(np.mean(block_img[block_mask]))
            block_means.append(xi_mean)

    Nb = len(block_means)
    if Nb == 0:
        return 0, 0.0

    x_mean = float(g_mean)
    block_means_array = np.asarray(block_means, dtype=np.float64)

    # (1/Nb) * Σ (xi_mean - x_mean)^2
    block_variance = float(np.mean((block_means_array - x_mean) ** 2))

    # Q7 = round( (1 - (1/121) * sqrt(variance)) * 100 )
    sqrt_variance = np.sqrt(block_variance)
    q7_raw = (1.0 - (1.0 / 121.0) * sqrt_variance) * 100.0
    Q7_score = int(round(q7_raw))
    Q7_score = max(0, min(100, Q7_score))

    return Q7_score, block_variance

