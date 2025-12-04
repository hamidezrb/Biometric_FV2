"""
ISO/IEC 29794-9 Quality Metrics - Shared Utilities
Finger Vascular Biometrics - Second Phalanx

This module contains shared utility functions for quality metric testing,
including test image creation functions.

Note: Individual quality components are implemented in separate files:
- q1.py: Q1 (Effective Area) implementation
- q3.py: Q3 (Contrast) implementation
"""

import cv2
import numpy as np
from typing import Tuple


def create_test_image(width: int, height: int, foreground_pixels: int, output_path: str) -> None:
    """
    Create a test image with a specified number of foreground pixels.
    
    Args:
        width (int): Image width
        height (int): Image height  
        foreground_pixels (int): Number of foreground pixels to create
        output_path (str): Path to save the test image
    """
    
    # Create black background
    image = np.zeros((height, width), dtype=np.uint8)
    
    # Calculate how many pixels we can actually fit
    max_pixels = width * height
    actual_pixels = min(foreground_pixels, max_pixels)
    
    if actual_pixels > 0:
        # Create a rectangular region of foreground pixels
        # Calculate dimensions for a roughly square region
        side_length = int(np.sqrt(actual_pixels))
        if side_length * side_length < actual_pixels:
            side_length += 1
            
        # Ensure we don't exceed image boundaries
        side_length = min(side_length, width, height)
        
        # Center the region in the image
        start_x = (width - side_length) // 2
        start_y = (height - side_length) // 2
        
        # Fill the region with white (foreground)
        end_x = min(start_x + side_length, width)
        end_y = min(start_y + side_length, height)
        
        image[start_y:end_y, start_x:end_x] = 255
        
        # If we need more pixels, add them in a systematic way
        remaining_pixels = actual_pixels - (end_x - start_x) * (end_y - start_y)
        if remaining_pixels > 0:
            # Add remaining pixels row by row
            for y in range(start_y, min(start_y + side_length, height)):
                for x in range(start_x, min(start_x + side_length, width)):
                    if remaining_pixels <= 0:
                        break
                    if image[y, x] == 0:  # Only fill empty pixels
                        image[y, x] = 255
                        remaining_pixels -= 1
                if remaining_pixels <= 0:
                    break
    
    # Save the test image
    cv2.imwrite(output_path, image)
    print(f"Created test image: {output_path} with {actual_pixels} foreground pixels")


def create_high_contrast_image(width: int, height: int, output_path: str) -> None:
    """
    Create a high-contrast test image for Q3 testing.
    Uses a checkerboard pattern with distinct gray values (0 and 255) to maximize contrast.
    Creates a foreground region that ensures Q1 mask covers the entire high-contrast area.
    
    Args:
        width (int): Image width
        height (int): Image height
        output_path (str): Path to save the test image
    """
    # Create image with black background
    image = np.zeros((height, width), dtype=np.uint8)
    
    # Create a foreground region (white border/frame) that contains the high-contrast pattern
    # This ensures Q1's mask will capture the entire region
    margin = min(width, height) // 8
    start_x = margin
    end_x = width - margin
    start_y = margin
    end_y = height - margin
    
    # Fill the entire foreground region with white first (so Q1 detects it)
    image[start_y:end_y, start_x:end_x] = 255
    
    # Now create high-contrast checkerboard pattern within this region
    # Use alternating black and white squares
    pattern_size = 8  # Size of each checker square
    for y in range(start_y, end_y, pattern_size):
        for x in range(start_x, end_x, pattern_size):
            # Alternate between black (0) and white (255)
            if ((x - start_x) // pattern_size + (y - start_y) // pattern_size) % 2 == 0:
                image[y:min(y+pattern_size, end_y), x:min(x+pattern_size, end_x)] = 0
    
    cv2.imwrite(output_path, image)
    print(f"Created high-contrast test image: {output_path}")


def create_low_contrast_image(width: int, height: int, output_path: str, gray_value: int = 128) -> None:
    """
    Create a low-contrast (near-uniform) test image for Q3 testing.
    Uses a uniform gray value to minimize contrast.
    
    Args:
        width (int): Image width
        height (int): Image height
        output_path (str): Path to save the test image
        gray_value (int): Gray value for uniform image (default: 128)
    """
    # Create uniform gray image
    image = np.full((height, width), gray_value, dtype=np.uint8)
    
    cv2.imwrite(output_path, image)
    print(f"Created low-contrast test image: {output_path} with gray value: {gray_value}")


def create_high_noise_suppression_image(width: int, height: int, output_path: str, gray_value: int = 128, noise_level: float = 0.01) -> None:
    """
    Create a high noise suppression (low variance) test image for Q4 testing.
    Uses a nearly uniform gray value with minimal noise to maximize Q4 (low sigma/g_mean ratio).
    
    Args:
        width (int): Image width
        height (int): Image height
        output_path (str): Path to save the test image
        gray_value (int): Base gray value (default: 128)
        noise_level (float): Noise level (0.0-1.0, default: 0.01 for minimal noise)
    """
    # Create uniform gray image with minimal noise
    image = np.full((height, width), gray_value, dtype=np.uint8).astype(np.float32)
    
    # Add minimal noise
    noise = np.random.normal(0, noise_level * 255, (height, width))
    image = np.clip(image + noise, 0, 255).astype(np.uint8)
    
    cv2.imwrite(output_path, image)
    print(f"Created high noise suppression test image: {output_path} (gray: {gray_value}, noise: {noise_level})")


def create_low_noise_suppression_image(width: int, height: int, output_path: str) -> None:
    """
    Create a low noise suppression (high variance) test image for Q4 testing.
    Uses an extreme high-contrast pattern to maximize variance (high sigma/g_mean ratio > 2).
    This creates a ratio that should result in Q4 < 20.
    
    Args:
        width (int): Image width
        height (int): Image height
        output_path (str): Path to save the test image
    """
    # Create image with very high variance (extreme contrast pattern)
    # To achieve Q4 < 20, we need sigma/g_mean ratio > 2
    image = np.zeros((height, width), dtype=np.uint8)
    
    # Create a foreground region with extreme high variance pattern
    margin = min(width, height) // 8
    start_x = margin
    end_x = width - margin
    start_y = margin
    end_y = height - margin
    
    # Fill with extreme alternating pattern (very high variance)
    # Use maximum contrast (0 and 255) to maximize variance
    # For 50% each: mean = 127.5, sigma ≈ 127.5, ratio ≈ 1.0
    # To get ratio > 2, we need an asymmetric pattern or use all pixels
    # Let's use 0 and 255 with equal distribution for maximum variance
    pattern_size = 2  # Smaller pattern for more variation
    for y in range(start_y, end_y, pattern_size):
        for x in range(start_x, end_x, pattern_size):
            # Alternate between black (0) and white (255) for maximum variance
            if ((x - start_x) // pattern_size + (y - start_y) // pattern_size) % 2 == 0:
                image[y:min(y+pattern_size, end_y), x:min(x+pattern_size, end_x)] = 0
            else:
                image[y:min(y+pattern_size, end_y), x:min(x+pattern_size, end_x)] = 255
    
    cv2.imwrite(output_path, image)
    print(f"Created low noise suppression test image: {output_path}")


def create_perfect_uniformity_image(width: int, height: int, output_path: str, gray_value: int = 128) -> None:
    """
    Create a perfectly uniform image for Q7 testing (TEST K).
    Uses a single shade of gray across the entire foreground region to achieve Q7 = 100.
    
    Args:
        width (int): Image width
        height (int): Image height
        output_path (str): Path to save the test image
        gray_value (int): Uniform gray value (default: 128)
    """
    # Create image with uniform gray value
    image = np.full((height, width), gray_value, dtype=np.uint8)
    
    # Create a foreground region (white border/frame) that ensures Q1 mask covers the entire uniform area
    margin = min(width, height) // 8
    start_x = margin
    end_x = width - margin
    start_y = margin
    end_y = height - margin
    
    # Fill the entire foreground region with uniform gray
    image[start_y:end_y, start_x:end_x] = gray_value
    
    cv2.imwrite(output_path, image)
    print(f"Created perfect uniformity test image: {output_path} (gray: {gray_value})")


def create_poor_uniformity_image(width: int, height: int, output_path: str) -> None:
    """
    Create a poor uniformity (hot spot) image for Q7 testing (TEST L).
    Uses a bright corner/block that is significantly brighter than the rest to achieve Q7 = 0-30.
    
    Args:
        width (int): Image width
        height (int): Image height
        output_path (str): Path to save the test image
    """
    # Create image with black background
    image = np.zeros((height, width), dtype=np.uint8)
    
    # Create a large foreground region (white border/frame) that ensures Q1 mask covers the entire area
    # Use smaller margin to create larger foreground region
    margin = min(width, height) // 10
    start_x = margin
    end_x = width - margin
    start_y = margin
    end_y = height - margin
    
    # Fill the entire foreground region with base gray (bright enough for Q1 to detect)
    # Use a value that's clearly above threshold (e.g., 100) but still much darker than hot spot
    base_gray = 100
    image[start_y:end_y, start_x:end_x] = base_gray
    
    # Create a bright hot spot in one corner (top-left corner of foreground region)
    # Make it significantly brighter to create high variance in block means
    # Use a large hot spot to ensure it affects multiple 5x5 blocks
    hot_spot_size = min(width, height) // 2  # Large hot spot (half the image)
    hot_spot_x = start_x
    hot_spot_y = start_y
    hot_spot_end_x = min(start_x + hot_spot_size, end_x)
    hot_spot_end_y = min(start_y + hot_spot_size, end_y)
    
    # Fill hot spot with very bright value (255 for maximum contrast)
    # This creates a large difference between hot spot blocks (mean ~255) and base gray blocks (mean ~100)
    # The variance will be: many blocks with mean ~100, many blocks with mean ~255
    # This should create high variance in block means, resulting in low Q7 (< 30)
    image[hot_spot_y:hot_spot_end_y, hot_spot_x:hot_spot_end_x] = 255
    
    cv2.imwrite(output_path, image)
    print(f"Created poor uniformity (hot spot) test image: {output_path}")


def create_high_sharpness_image(width: int, height: int, output_path: str) -> None:
    """
    Create a high-sharpness synthetic image for Q6 testing (TEST K).
    Generates crisp edges and diagonals that should yield a high N100 count.
    """
    image = np.zeros((height, width), dtype=np.uint8)

    # Draw bright rectangle with sharp edges
    margin = min(width, height) // 6
    top_left = (margin, margin)
    bottom_right = (width - margin, height - margin)
    cv2.rectangle(image, top_left, bottom_right, 255, thickness=3)

    # Add diagonal lines for additional sharp features
    cv2.line(image, top_left, bottom_right, 255, thickness=2)
    cv2.line(image, (margin, height - margin), (width - margin, margin), 255, thickness=2)

    cv2.imwrite(output_path, image)
    print(f"Created high sharpness test image: {output_path}")


def create_blurred_image(width: int, height: int, output_path: str) -> None:
    """
    Create a blurred (low-sharpness) image for Q6 testing (TEST L).
    """
    image = np.zeros((height, width), dtype=np.uint8)

    margin = min(width, height) // 4
    cv2.rectangle(image, (margin, margin), (width - margin, height - margin), 200, thickness=-1)

    blurred = cv2.GaussianBlur(image, (31, 31), sigmaX=12, sigmaY=12)
    cv2.imwrite(output_path, blurred)
    print(f"Created blurred test image: {output_path}")


def create_high_vessel_density_image(width: int, height: int, output_path: str, target_pixels: int = 800) -> None:
    """
    Create a high vessel density image for Q8 testing (TEST M).
    Generates a pattern with many thin lines that will result in high N_vessel after skeletonization.
    
    Args:
        width (int): Image width
        height (int): Image height
        output_path (str): Path to save the test image
        target_pixels (int): Target number of vessel pixels after skeletonization (default: 800)
    """
    image = np.zeros((height, width), dtype=np.uint8)
    
    # Create a foreground region
    margin = min(width, height) // 8
    start_x = margin
    end_x = width - margin
    start_y = margin
    end_y = height - margin
    
    # Fill foreground region with base gray value (bright enough for Q1 to detect)
    image[start_y:end_y, start_x:end_x] = 200
    
    # Draw multiple thin lines (vessels) to create high density
    # Use a grid pattern with many intersecting lines
    line_spacing = 8
    line_thickness = 1
    
    # Horizontal lines
    for y in range(start_y, end_y, line_spacing):
        cv2.line(image, (start_x, y), (end_x, y), 255, line_thickness)
    
    # Vertical lines
    for x in range(start_x, end_x, line_spacing):
        cv2.line(image, (x, start_y), (x, end_y), 255, line_thickness)
    
    # Diagonal lines for additional density
    for offset in range(0, min(end_x - start_x, end_y - start_y), line_spacing * 2):
        cv2.line(image, (start_x + offset, start_y), 
                (min(start_x + offset + (end_y - start_y), end_x), end_y), 255, line_thickness)
        cv2.line(image, (start_x, start_y + offset),
                (end_x, min(start_y + offset + (end_x - start_x), end_y)), 255, line_thickness)
    
    cv2.imwrite(output_path, image)
    print(f"Created high vessel density test image: {output_path} (target pixels: {target_pixels})")


def create_low_vessel_density_image(width: int, height: int, output_path: str, target_pixels: int = 150) -> None:
    """
    Create a low vessel density image for Q8 testing (TEST N).
    Generates a sparse pattern with few lines that will result in low N_vessel after skeletonization.
    
    Args:
        width (int): Image width
        height (int): Image height
        output_path (str): Path to save the test image
        target_pixels (int): Target number of vessel pixels after skeletonization (default: 150)
    """
    image = np.zeros((height, width), dtype=np.uint8)
    
    # Create a foreground region
    margin = min(width, height) // 8
    start_x = margin
    end_x = width - margin
    start_y = margin
    end_y = height - margin
    
    # Fill foreground region with base gray value (bright enough for Q1 to detect)
    image[start_y:end_y, start_x:end_x] = 200
    
    # Draw sparse lines (vessels) to create low density
    # Use wide spacing between lines
    line_spacing = 40
    line_thickness = 1
    
    # Only a few horizontal lines
    for y in range(start_y, end_y, line_spacing):
        if y < end_y:
            cv2.line(image, (start_x, y), (end_x, y), 255, line_thickness)
    
    # Only a few vertical lines
    for x in range(start_x, end_x, line_spacing):
        if x < end_x:
            cv2.line(image, (x, start_y), (x, end_y), 255, line_thickness)
    
    cv2.imwrite(output_path, image)
    print(f"Created low vessel density test image: {output_path} (target pixels: {target_pixels})")


def _sobel_4dir(gray: np.ndarray) -> np.ndarray:
    """
    Compute Sobel responses in four directions (0°, 45°, 90°, 135°) and average magnitudes.
    """
    if gray.dtype != np.float32:
        gray_f = gray.astype(np.float32)
    else:
        gray_f = gray

    gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)

    k45 = np.array([[0, 1, 2],
                    [-1, 0, 1],
                    [-2, -1, 0]], dtype=np.float32)
    k135 = np.array([[2, 1, 0],
                     [1, 0, -1],
                     [0, -1, -2]], dtype=np.float32)

    g45 = cv2.filter2D(gray_f, cv2.CV_32F, k45)
    g135 = cv2.filter2D(gray_f, cv2.CV_32F, k135)

    w_mean = (np.abs(gx) + np.abs(gy) + np.abs(g45) + np.abs(g135)) / 4.0
    return w_mean


def calculate_q6(R_mask: np.ndarray,
                 Grayscale_Image: np.ndarray,
                 S_unoccluded: int,
                 gc: float = 0.006,
                 threshold: int = 100) -> Tuple[int, int]:
    """
    Calculate Q6 (Sharpness) following ISO/IEC 29794-9 Clause 5.2.6.

    Returns (Q6_score, N100) where N100 is the count of normalized edge responses > threshold.
    """
    foreground = (R_mask == 255)
    if S_unoccluded <= 0 or np.count_nonzero(foreground) == 0:
        return 0, 0

    gray = Grayscale_Image
    if gray.dtype != np.uint8:
        gray = gray.astype(np.uint8)

    w_mean = _sobel_4dir(gray)
    min_val = float(w_mean.min())
    max_val = float(w_mean.max())
    if max_val > min_val:
        w_norm = (w_mean - min_val) * (255.0 / (max_val - min_val))
    else:
        w_norm = np.zeros_like(w_mean, dtype=np.float32)

    strong_edges_mask = (w_norm > threshold) & foreground
    N100 = int(np.count_nonzero(strong_edges_mask))

    denominator = gc * S_unoccluded
    if denominator <= 0:
        return 0, N100

    q6_raw = (N100 / denominator) * 100.0
    Q6_score = int(round(q6_raw))
    Q6_score = max(0, min(100, Q6_score))
    return Q6_score, N100


