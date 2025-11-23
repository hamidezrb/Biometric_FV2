"""
ISO/IEC 29794-9 Quality Component 3 (Contrast) Implementation
Finger Vascular Biometrics - Second Phalanx

This module implements the mandatory ISO calculation for Q3 (Contrast)
according to ISO/IEC 29794-9 Clause 5.2.3.

Requirements:
- Standard Deviation (Sigma): Calculate using Formula (6)
- Normalization: Use 2^D/4 as denominator (for 8-bit: 2^8/4 = 64)
- Formula: Q3 = ROUND( (Sigma / (2^D / 4)) * 100 )

Note: Q3 and Q4 share the same statistical inputs (sigma and g_mean),
so calculate_q3() returns both Q3 and Q4 scores together.
"""

"""Contrast(Low contrast in vascular images indicates poor image quality.
When the contrast of vascular images is low, the differentiation between 
vascular and non-vascular areas is reduced, leading to blurred or absent vascular patterns.)"""

import numpy as np
from typing import Tuple
from q4 import calculate_q4
from quality_metrics import calculate_q6


def calculate_q3(R_mask: np.ndarray,
                 Grayscale_Image: np.ndarray,
                 S_unoccluded: int,
                 bit_depth: int = 8,
                 include_q4: bool = True,
                 include_q6: bool = True,
                 gc: float = 0.006) -> Tuple[int, int, int, float, float, int]:
    """
    Calculate Q3 (Contrast), Q4 (Equivalent Number of Looks), and Q6 (Sharpness).
    These metrics share the same statistical inputs (sigma and g_mean).
    
    Args:
        R_mask (np.ndarray): Binary mask of foreground region (255 for foreground, 0 for background)
        Grayscale_Image (np.ndarray): Original grayscale image
        S_unoccluded (int): Foreground pixel count (Sunoccluded) from Q1
        bit_depth (int): Bit depth of the image (default: 8 for 8-bit images)
        include_q4 (bool): Whether to calculate Q4
        include_q6 (bool): Whether to calculate Q6
        gc (float): Scaling coefficient for Q6 (default 0.006)
        
    Returns:
        Tuple[int, int, int, float, float, int]:
            (Q3_score, Q4_score, Q6_score, sigma, g_mean, N100)
            - Q3_score: Final integer Q3 score (0-100)
            - Q4_score: Final integer Q4 score (0-100)
            - Q6_score: Final integer Q6 score (0-100)
            - sigma: Standard deviation of pixel values within R_mask
            - g_mean: Mean grayscale value within R_mask
            - N100: Strong edge count used in Q6
    
    ISO Requirements [Clause 5.2.3, 5.2.4, and 5.2.6]:
    Q3:
    1. Veto Logic: If R_mask is invalid (area=0), return Q3 = 0
    2. Standard Deviation (Sigma): Calculate using Formula (6)
    3. Normalization: Use 2^D/4 as denominator (for 8-bit: 2^8/4 = 64)
    4. Q3 Calculation using Formula (7): Q3 = ROUND( (Sigma / (2^D / 4)) * 100 )
    
    Q4:
    1. Veto Logic: If g_mean is zero, return Q4 = 0
    2. Q4 Calculation using Formula (8): Q4 = MIN(100, ROUND( 1 / (1 + (sigma/g_mean)^2) * 100 ) )

    Q6:
    1. Veto Logic: If S_unoccluded is zero, return Q6 = 0
    2. Uses Sobel filters in four directions to compute W_mean
    3. Normalize W_mean, count N100 (pixels > 100) inside R_mask
    4. Q6 Calculation using Formula (12): Q6 = MIN(100, ROUND( N100 / (gc * S_unoccluded) * 100 ))
    """
    
    # Veto Logic: If R_mask is invalid (area=0), return Q3 = 0, Q4 = 0
    # Get foreground pixels (where mask is 255)
    foreground_mask = (R_mask == 255)
    #Find all the white pixels — that’s the finger.
    N = np.sum(foreground_mask)
    
    if N == 0:
        return 0, 0, 0, 0.0, 0.0, 0
    
    # Extract pixel values within the foreground region
    ##all the gray colors inside the finger
    foreground_pixels = Grayscale_Image[foreground_mask]
    
    # What is the average color of the finge
    g_mean = np.mean(foreground_pixels)
    
    # Calculate standard deviation (Sigma) using Formula (6)
    # Sigma = SQRT( (1/N) * SUM( (gi - g_mean)^2 ) )
    # This is equivalent to np.std() with ddof=0
    squared_diff = (foreground_pixels - g_mean) ** 2
    variance = np.sum(squared_diff) / N
    sigma = np.sqrt(variance)
    
    # Calculate Q3
    # Normalization: Use 2^D/4 as denominator
    # For 8-bit: 2^8/4 = 256/4 = 64
    normalization_factor = (2 ** bit_depth) / 4
    
    # Q3 Calculation using Formula (7): Q3 = ROUND( (Sigma / (2^D / 4)) * 100 )
    q3_raw = (sigma / normalization_factor) * 100
    Q3_score = round(q3_raw)
    # The larger the score of the contrast component, the higher the image quality.
    Q3_score = min(100, max(0, Q3_score))
    
    # Calculate Q4 using the shared sigma and g_mean (if requested)
    if include_q4:
        Q4_score = calculate_q4(sigma, g_mean)
    else:
        Q4_score = 0
    
    if include_q6:
        Q6_score, N100 = calculate_q6(R_mask, Grayscale_Image, S_unoccluded, gc=gc)
    else:
        Q6_score, N100 = 0, 0
    
    return Q3_score, Q4_score, Q6_score, sigma, g_mean, N100

