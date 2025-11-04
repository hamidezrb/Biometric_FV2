"""
ISO/IEC 29794-9 Quality Component 4 (Equivalent Number of Looks - ENL) Implementation
Finger Vascular Biometrics - Second Phalanx

This module implements the mandatory ISO calculation for Q4 (Equivalent Number of Looks - ENL)
according to ISO/IEC 29794-9 Clause 5.2.4.

Requirements:
- Input: Standard Deviation (sigma) and Mean Grayscale Value (g_mean)
- Veto Logic: If g_mean is zero, return Q4 = 0
- Formula: Q4 = MIN(100, ROUND( 1 / (1 + (sigma/g_mean)^2) * 100 ) )
"""

from typing import Tuple


def calculate_q4(sigma: float, g_mean: float) -> int:
    """
    Calculate Q4 (Equivalent Number of Looks - ENL) quality metric.
    
    Args:
        sigma (float): Standard deviation of pixel values
        g_mean (float): Mean grayscale value
        
    Returns:
        int: Q4_score - Final integer Q4 score (0-100)
    
    ISO Requirements [Clause 5.2.4]:
    1. Veto Logic: If g_mean is zero (which occurs if area N=0), Q4 = 0
    2. Q4 Calculation using Formula (8): Q4 = MIN(100, ROUND( 1 / (1 + (sigma/g_mean)^2) * 100 ) )
    3. Q4 must be integer (0-100)
    """
    
    # Veto Logic: If g_mean is zero, return Q4 = 0
    if g_mean == 0.0:
        return 0
    
    # Avoid division by zero (shouldn't happen if g_mean != 0, but safety check)
    if sigma == 0.0:
        # If sigma is zero, the ratio is 0, so Q4 = 100 (perfect uniformity)
        return 100
    
    # Calculate sigma/g_mean ratio
    ratio = sigma / g_mean
    
    # Q4 Calculation using Formula (8): Q4 = MIN(100, ROUND( 1 / (1 + (sigma/g_mean)^2) * 100 ) )
    q4_raw = (1 / (1 + ratio ** 2)) * 100
    Q4_score = round(q4_raw)
    
    # Cap at 100
    Q4_score = min(100, max(0, Q4_score))
    
    return Q4_score

