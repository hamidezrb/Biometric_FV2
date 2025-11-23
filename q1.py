"""
ISO/IEC 29794-9 Quality Component 1 (Effective Area) Implementation
Finger Vascular Biometrics - Second Phalanx

This module implements the mandatory ISO calculation for Q1 (Effective Area)
according to ISO/IEC 29794-9 Clause 5.2.1.

Requirements:
- Coefficient (Sc): 20000 pixels for Finger
- Veto Logic: If Foreground Region (R) is invalid (zero pixels), Q1 = 0
- Formula: Q1 = MIN(100, ROUND(Sunoccluded / Sc * 100))
"""

import cv2
import numpy as np
from typing import Tuple


def calculate_q1(image_path: str) -> Tuple[int, int, np.ndarray, np.ndarray]:
    """
    Calculate Q1 (Effective Area) quality metric for finger vascular biometrics.
    Refactored to return intermediate data for Q3 calculation.
    
    Args:
        image_path (str): Path to the input image file
        
    Returns:
        Tuple[int, int, np.ndarray, np.ndarray]: 
            - Q1_score: Final integer Q1 score (0-100)
            - S_unoccluded: Raw pixel count of unoccluded foreground region
            - R_mask: Binary mask of foreground region (255 for foreground, 0 for background)
            - Grayscale_Image: Original grayscale image
    
    ISO Requirements:
    1. Coefficient (Sc) = 20000 pixels (for Finger)
    2. Veto Logic: If Foreground Region (R) is invalid (zero pixels), Q1 = 0
    3. Formula: Q1 = MIN(100, ROUND(Sunoccluded / Sc * 100))
    """
    
    # ISO Coefficient for Finger (Clause 5.2.1)
    Sc = 20000  # pixels , Effective Area Standard Area = 20,000 pixels
    
    try:
        # Load the image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Automatic thresholding
        # Produces a binary mask: 0 background, 255 foreground
        _, binary_mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours to identify the foreground region
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            # No contours found - invalid foreground region
            # Return zero mask and original image
            zero_mask = np.zeros_like(image)
            return 0, 0, zero_mask, image
        
        # Find the largest contour (main foreground region) (Finds the biggest shape)
        #Assumes the largest object is the finger
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Create mask for the largest contour (R_mask)
        R_mask = np.zeros_like(binary_mask)
        cv2.fillPoly(R_mask, [largest_contour], 255)#draw the finger shape in pure white
        

        # Calculate S_unoccluded (pixel count of unoccluded foreground region)
        #How many pixels are in the finger?”
        S_unoccluded = np.sum(R_mask == 255)
        
        # Veto Logic: If Foreground Region (R) is invalid (zero pixels), Q1 = 0
        if S_unoccluded == 0:
            return 0, 0, R_mask, image
        
        # ISO Formula: Q1 = MIN(100, ROUND(Sunoccluded / Sc * 100))
        q1_raw = S_unoccluded / Sc * 100
        Q1_score = min(100, round(q1_raw))
        
        return Q1_score, S_unoccluded, R_mask, image
        
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        # Return zero mask and empty image on error
        zero_mask = np.zeros((100, 100), dtype=np.uint8)
        zero_image = np.zeros((100, 100), dtype=np.uint8)
        return 0, 0, zero_mask, zero_image

