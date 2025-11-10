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

