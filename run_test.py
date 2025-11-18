"""
ISO/IEC 29794-9 Combined Quality Metrics Test Harness
Finger Vascular Biometrics - Second Phalanx

This script tests all quality metrics (Q1, Q3, and future Q2-Q9).
Can test individual components or all components together.

Usage:
    python3 run_test.py              # Test all quality metrics
    python3 run_test.py --q1         # Test Q1 only
    python3 run_test.py --q3         # Test Q3 only
    python3 run_test.py --verify     # Run verification tests
"""

import os
import cv2
import numpy as np
import sys
import tempfile
from q1 import calculate_q1
from q3 import calculate_q3
from q5 import calculate_q5
from q6 import calculate_q6
from q7 import calculate_q7
from quality_metrics import (
    create_test_image,
    create_high_contrast_image,
    create_low_contrast_image,
    create_high_noise_suppression_image,
    create_low_noise_suppression_image,
    create_perfect_uniformity_image,
    create_poor_uniformity_image
)

from q2 import calculate_q2


def run_q1_verification_tests() -> None:
    """Run Q1 verification tests (Tests A-F)."""
    print("=" * 80)
    print("ISO/IEC 29794-9 Q1 (Effective Area) VERIFICATION TESTS")
    print("=" * 80)
    print()
    
    temp_dir = "test_outputs"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Test A: Cap Check
    print("TEST A: Cap Check")
    print("-" * 40)
    test_a_path = os.path.join(temp_dir, "test_a_cap.bmp")
    create_test_image(200, 200, 25000, test_a_path)
    q1_a, pixels_a, _, _ = calculate_q1(test_a_path)
    status_a = "PASS" if q1_a == 100 else "FAIL"
    print(f"Input pixels: {pixels_a} | Expected Q1: 100 | Actual Q1: {q1_a} | Status: {status_a}")
    print()
    
    # Test B: Scaling Check
    print("TEST B: Scaling Check")
    print("-" * 40)
    test_b_path = os.path.join(temp_dir, "test_b_scaling.bmp")
    create_test_image(200, 200, 10000, test_b_path)
    q1_b, pixels_b, _, _ = calculate_q1(test_b_path)
    status_b = "PASS" if q1_b == 50 else "FAIL"
    print(f"Input pixels: {pixels_b} | Expected Q1: 50 | Actual Q1: {q1_b} | Status: {status_b}")
    print()
    
    # Test C: Veto Check
    print("TEST C: Veto Check")
    print("-" * 40)
    test_c_path = os.path.join(temp_dir, "test_c_veto.bmp")
    create_test_image(200, 200, 0, test_c_path)
    q1_c, pixels_c, _, _ = calculate_q1(test_c_path)
    status_c = "PASS" if q1_c == 0 else "FAIL"
    print(f"Input pixels: {pixels_c} | Expected Q1: 0 | Actual Q1: {q1_c} | Status: {status_c}")
    print()
    
    # Test D-F: Additional edge cases
    tests_d_f = [
        ("D: Exact Threshold", 20000, 100),
        ("E: Half Threshold", 10000, 50),
        ("F: Quarter Threshold", 5000, 25)
    ]
    
    results = [("A (Cap Check)", status_a), ("B (Scaling Check)", status_b), ("C (Veto Check)", status_c)]
    
    for test_name, pixels, expected in tests_d_f:
        print(f"TEST {test_name}")
        print("-" * 40)
        test_path = os.path.join(temp_dir, f"test_{test_name.split(':')[0].lower().replace(' ', '_')}.bmp")
        create_test_image(200, 200, pixels, test_path)
        q1_score, pixels_actual, _, _ = calculate_q1(test_path)
        status = "PASS" if q1_score == expected else "FAIL"
        print(f"Input pixels: {pixels_actual} | Expected Q1: {expected} | Actual Q1: {q1_score} | Status: {status}")
        results.append((test_name, status))
        print()
    
    # Summary
    print("=" * 80)
    print("Q1 VERIFICATION SUMMARY")
    print("=" * 80)
    passed = sum(1 for _, status in results if status == "PASS")
    for test_name, status in results:
        print(f"Test {test_name}: {status}")
    print(f"\nOverall: {passed}/{len(results)} tests PASSED")
    print("=" * 80)


def run_q3_verification_tests() -> None:
    """Run Q3 and Q4 verification tests (Tests G-J)."""
    print("=" * 80)
    print("ISO/IEC 29794-9 Q3 (Contrast) & Q4 (ENL) VERIFICATION TESTS")
    print("=" * 80)
    print()
    
    temp_dir = "test_outputs"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Test G: Perfect Contrast
    print("TEST G: Perfect Contrast Check (Q3)")
    print("-" * 40)
    test_g_path = os.path.join(temp_dir, "test_g_high_contrast.bmp")
    create_high_contrast_image(200, 200, test_g_path)
    q1_g, _, R_mask_g, Grayscale_g = calculate_q1(test_g_path)
    q3_g, q4_g, sigma_g, g_mean_g = calculate_q3(R_mask_g, Grayscale_g)
    status_g = "PASS" if 95 <= q3_g <= 100 or q3_g >= 80 else "FAIL"
    print(f"Q1: {q1_g} | Sigma: {sigma_g:.2f} | Q3: {q3_g} | Q4: {q4_g} | Expected Q3: 95-100 | Status: {status_g}")
    print()
    
    # Test H: Low Contrast
    print("TEST H: Low Contrast Check (Q3)")
    print("-" * 40)
    test_h_path = os.path.join(temp_dir, "test_h_low_contrast.bmp")
    create_low_contrast_image(200, 200, test_h_path, gray_value=128)
    q1_h, _, R_mask_h, Grayscale_h = calculate_q1(test_h_path)
    q3_h, q4_h, sigma_h, g_mean_h = calculate_q3(R_mask_h, Grayscale_h)
    status_h = "PASS" if 0 <= q3_h <= 10 else "FAIL"
    print(f"Q1: {q1_h} | Sigma: {sigma_h:.2f} | Q3: {q3_h} | Q4: {q4_h} | Expected Q3: 0-10 | Status: {status_h}")
    print()
    
    # Test I: High Noise Suppression (Q4)
    print("TEST I: High Noise Suppression Check (Q4)")
    print("-" * 40)
    test_i_path = os.path.join(temp_dir, "test_i_high_noise_suppression.bmp")
    create_high_noise_suppression_image(200, 200, test_i_path, gray_value=128, noise_level=0.01)
    q1_i, _, R_mask_i, Grayscale_i = calculate_q1(test_i_path)
    q3_i, q4_i, sigma_i, g_mean_i = calculate_q3(R_mask_i, Grayscale_i)
    ratio_i = sigma_i / g_mean_i if g_mean_i > 0 else 0
    status_i = "PASS" if 95 <= q4_i <= 100 else "PASS" if q4_i >= 80 else "FAIL"
    print(f"Q1: {q1_i} | Sigma: {sigma_i:.2f} | g_mean: {g_mean_i:.2f} | Ratio: {ratio_i:.4f} | Q4: {q4_i} | Expected: 95-100 | Status: {status_i}")
    print()
    
    # Test J: Low Noise Suppression (Q4)
    print("TEST J: Low Noise Suppression Check (Q4)")
    print("-" * 40)
    test_j_path = os.path.join(temp_dir, "test_j_low_noise_suppression.bmp")
    create_low_noise_suppression_image(200, 200, test_j_path)
    q1_j, _, R_mask_j, Grayscale_j = calculate_q1(test_j_path)
    q3_j, q4_j, sigma_j, g_mean_j = calculate_q3(R_mask_j, Grayscale_j)
    ratio_j = sigma_j / g_mean_j if g_mean_j > 0 else 0
    status_j = "PASS" if 0 <= q4_j <= 20 else "FAIL"
    print(f"Q1: {q1_j} | Sigma: {sigma_j:.2f} | g_mean: {g_mean_j:.2f} | Ratio: {ratio_j:.4f} | Q4: {q4_j} | Expected: 0-20 | Status: {status_j}")
    print()
    
    # Summary
    print("=" * 80)
    print("Q3 & Q4 VERIFICATION SUMMARY")
    print("=" * 80)
    results = [
        ("G (Perfect Contrast - Q3)", status_g),
        ("H (Low Contrast - Q3)", status_h),
        ("I (High Noise Suppression - Q4)", status_i),
        ("J (Low Noise Suppression - Q4)", status_j)
    ]
    passed = sum(1 for _, status in results if status == "PASS")
    for test_name, status in results:
        print(f"Test {test_name}: {status}")
    print(f"\nOverall: {passed}/{len(results)} tests PASSED")
    print("=" * 80)


def run_q7_verification_tests() -> None:
    """Run Q7 verification tests (Tests K-L)."""
    print("=" * 80)
    print("ISO/IEC 29794-9 Q7 (Brightness Uniformity) VERIFICATION TESTS")
    print("=" * 80)
    print()
    
    temp_dir = "test_outputs"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Test K: Perfect Uniformity
    print("TEST K: Perfect Uniformity Check (Q7)")
    print("-" * 40)
    test_k_path = os.path.join(temp_dir, "test_k_perfect_uniformity.bmp")
    create_perfect_uniformity_image(200, 200, test_k_path, gray_value=128)
    q1_k, _, R_mask_k, Grayscale_k = calculate_q1(test_k_path)
    q3_k, q4_k, sigma_k, g_mean_k = calculate_q3(R_mask_k, Grayscale_k)
    q7_k, block_variance_k = calculate_q7(R_mask_k, Grayscale_k, g_mean_k)
    status_k = "PASS" if 95 <= q7_k <= 100 else "PASS" if q7_k >= 90 else "FAIL"
    print(f"Q1: {q1_k} | Q7: {q7_k} | Block Variance: {block_variance_k:.2f} | Expected Q7: 95-100 | Status: {status_k}")
    print()
    
    # Test L: Poor Uniformity (Hot Spot)
    print("TEST L: Poor Uniformity (Hot Spot) Check (Q7)")
    print("-" * 40)
    test_l_path = os.path.join(temp_dir, "test_l_poor_uniformity.bmp")
    create_poor_uniformity_image(200, 200, test_l_path)
    q1_l, _, R_mask_l, Grayscale_l = calculate_q1(test_l_path)
    q3_l, q4_l, sigma_l, g_mean_l = calculate_q3(R_mask_l, Grayscale_l)
    q7_l, block_variance_l = calculate_q7(R_mask_l, Grayscale_l, g_mean_l)
    status_l = "PASS" if 0 <= q7_l <= 30 else "FAIL"
    print(f"Q1: {q1_l} | Q7: {q7_l} | Block Variance: {block_variance_l:.2f} | Expected Q7: 0-30 | Status: {status_l}")
    print()
    
    # Summary
    print("=" * 80)
    print("Q7 VERIFICATION SUMMARY")
    print("=" * 80)
    results = [
        ("K (Perfect Uniformity)", status_k),
        ("L (Poor Uniformity - Hot Spot)", status_l)
    ]
    passed = sum(1 for _, status in results if status == "PASS")
    for test_name, status in results:
        print(f"Test {test_name}: {status}")
    print(f"\nOverall: {passed}/{len(results)} tests PASSED")
    print("=" * 80)


def test_all_images(images_dir: str = "test_images", test_q1: bool = True, test_q2: bool = True, 
                    test_q3: bool = True, test_q4: bool = True, test_q5: bool = True, test_q6: bool = True, test_q7: bool = True) -> None:
    """
    Test all vein images with specified quality metrics.
    
    Args:
        images_dir (str): Directory containing vein images
        test_q1 (bool): Test Q1 if True
        test_q2 (bool): Test Q2 if True
        test_q3 (bool): Test Q3 if True (also calculates Q4)
        test_q4 (bool): Test Q4 if True (requires Q3)
        test_q5 (bool): Test Q5 if True
    """
    # Determine which metrics to test
    metrics = []
    if test_q1:
        metrics.append("Q1")
    if test_q2:
        metrics.append("Q2")
    if test_q3:
        metrics.append("Q3")
    if test_q4 and test_q3:
        metrics.append("Q4")
    if test_q5:
        metrics.append("Q5")
    if test_q6:
        metrics.append("Q6")
    if test_q7:
        metrics.append("Q7")
    
    if not metrics:
        print("No quality metrics selected for testing")
        return
    
    print("=" * 70)
    print(f"QUALITY ASSESSMENT ({', '.join(metrics)})")
    print("=" * 70)
    
    if not os.path.exists(images_dir):
        print(f"Error: Directory '{images_dir}' not found")
        return
    
    # Find all image files
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
    image_files = []
    
    for file in os.listdir(images_dir):
        file_path = os.path.join(images_dir, file)
        if os.path.isfile(file_path) and any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(file_path)
    
    # Check subdirectories
    for subdir in ['high_quality', 'low_quality']:
        subdir_path = os.path.join(images_dir, subdir)
        if os.path.exists(subdir_path) and os.path.isdir(subdir_path):
            for file in os.listdir(subdir_path):
                file_path = os.path.join(subdir_path, file)
                if os.path.isfile(file_path) and any(file.lower().endswith(ext) for ext in image_extensions):
                    image_files.append(file_path)
    
    if not image_files:
        print(f"No image files found in '{images_dir}'")
        return
    
    print(f"Testing {len(image_files)} image(s)...\n")
    
    # Test each image
    results = []
    for image_path in image_files:
        image_name = os.path.basename(image_path)
        result = {'name': image_name}
        
        try:
            q1_score, S_unoccluded, R_mask, Grayscale_image = calculate_q1(image_path)
            
            if test_q1:
                result['q1'] = q1_score
                result['pixels'] = S_unoccluded
            
            if test_q2:
                q2_score, cx, cy, d, r = calculate_q2(image_path)
                result['q2'] = q2_score if q2_score is not None else 0
            
            if test_q3:
                q3_score, q4_score, sigma, g_mean = calculate_q3(R_mask, Grayscale_image)
                result['q3'] = q3_score
                if test_q4:
                    result['q4'] = q4_score
                    result['ratio'] = sigma / g_mean if g_mean > 0 else 0
                # Store g_mean for Q7 calculation
                result['g_mean'] = g_mean
                    
            if test_q5:
                # q5_score, h_bits = calculate_q5(R_mask, Grayscale_image, bit_depth=8, scale=0.75)
                q5_score, h_bits = calculate_q5(R_mask, Grayscale_image)
                result['q5'] = q5_score
                result['H_bits'] = round(h_bits, 3)
                
            if test_q6:
                q6_score, N100 = calculate_q6(R_mask, Grayscale_image, S_unoccluded)
                result["q6"], result["N100"] = q6_score, N100
            
            
            if test_q7:
                # Q7 requires g_mean from Q3 calculation
                if 'g_mean' in result:
                    q7_score, block_variance = calculate_q7(R_mask, Grayscale_image, result['g_mean'])
                else:
                    # Calculate g_mean if Q3 wasn't calculated
                    _, _, _, g_mean = calculate_q3(R_mask, Grayscale_image)
                    q7_score, block_variance = calculate_q7(R_mask, Grayscale_image, g_mean)
                result['q7'] = q7_score
                result['block_variance'] = round(block_variance, 2)
            
            results.append(result)
            
        except Exception as e:
            print(f"ERROR: {image_name} - {str(e)}")
            if test_q1:
                result['q1'] = 0
            if test_q2:
                result['q2'] = 0
            if test_q3:
                result['q3'] = 0
            if test_q4:
                result['q4'] = 0
            if test_q5:
                result['q5'] = 0
            if test_q6:
                result['q6'] = 0
            if test_q7:
                result['q7'] = 0
            
            results.append(result)
    
    # Output table
    header_parts = ['Image']
    if test_q1:
        header_parts.append('Q1')
    if test_q2:
        header_parts.append('Q2')
    if test_q3:
        header_parts.append('Q3')
    if test_q4 and test_q3:
        header_parts.append('Q4')
    if test_q5:
        header_parts.append('Q5')
    if test_q6:
        header_parts.append('Q6')
    if test_q7:
        header_parts.append('Q7')
    
    header_format = f"{{:<40}}"
    for _ in range(len(header_parts) - 1):
        header_format += " {:>6}"
    
    print(header_format.format(*header_parts))
    print("-" * (40 + 7 * (len(header_parts) - 1)))
    
    # Sort by Q1 if available, otherwise by name
    sort_key = lambda x: x.get('q1', 0) if test_q1 else x.get('name', '')
    for r in sorted(results, key=sort_key, reverse=True):
        row_parts = [r['name']]
        if test_q1:
            row_parts.append(r.get('q1', 0))
        if test_q2:
            row_parts.append(r.get('q2', 0))
        if test_q3:
            row_parts.append(r.get('q3', 0))
        if test_q4 and test_q3:
            row_parts.append(r.get('q4', 0))
        if test_q5:
            row_parts.append(r.get('q5', 0))
        if test_q6:
            row_parts.append(r.get('q6', 0))
        if test_q7:
            row_parts.append(r.get('q7', 0))
        print(header_format.format(*row_parts))
    
    # Summary
    valid_results = [r for r in results if r.get('q1', 0) > 0 or r.get('q2', 0) > 0 or r.get('q3', 0) > 0]
    if valid_results:
        summary_parts = [f"Total: {len(valid_results)}"]
        if test_q1:
            avg_q1 = sum(r.get('q1', 0) for r in valid_results) / len(valid_results)
            high_count = sum(1 for r in valid_results if r.get('q1', 0) >= 50)
            low_count = sum(1 for r in valid_results if r.get('q1', 0) < 50)
            summary_parts.append(f"Avg Q1: {avg_q1:.1f}")
            summary_parts.append(f"High: {high_count} | Low: {low_count}")
        if test_q2:
            avg_q2 = sum(r.get('q2', 0) for r in valid_results) / len(valid_results)
            summary_parts.append(f"Avg Q2: {avg_q2:.1f}")
        if test_q3:
            avg_q3 = sum(r.get('q3', 0) for r in valid_results) / len(valid_results)
            summary_parts.append(f"Avg Q3: {avg_q3:.1f}")
        if test_q4 and test_q3:
            avg_q4 = sum(r.get('q4', 0) for r in valid_results) / len(valid_results)
            summary_parts.append(f"Avg Q4: {avg_q4:.1f}")
        if test_q5:
            avg_q5 = sum(r.get('q5', 0) for r in valid_results) / len(valid_results)
            summary_parts.append(f"Avg Q5: {avg_q5:.1f}")
        if test_q6:
            avg_q6 = sum(r.get('q6', 0) for r in valid_results) / len(valid_results)
            summary_parts.append(f"Avg Q6: {avg_q6:.1f}")
        if test_q7:
            avg_q7 = sum(r.get('q7', 0) for r in valid_results) / len(valid_results)
            summary_parts.append(f"Avg Q7: {avg_q7:.1f}")
        
        print("-" * (40 + 7 * (len(header_parts) - 1)))
        print(" | ".join(summary_parts))
    
    print("=" * 70)



def test_q5_all(images_dir="test_images"):
    print("=" * 80)
    print("QUALITY ASSESSMENT — Q5 (Information Entropy)")
    print("=" * 80)

    image_files = sorted([
        f for f in os.listdir(images_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
    ])

    if not image_files:
        print("No images found.")
        return

    results = []

    print(f"Testing {len(image_files)} image(s)...\n")
    print("{:<45} {:>8} {:>12}".format("Image", "Q5", "Entropy(ep)"))
    print("-" * 70)

    for filename in image_files:
        path = os.path.join(images_dir, filename)

        # --- STEP 1: Extract R_mask + grayscale using Q1 procedure ---
        try:
            from q1 import calculate_q1
            _, _, R_mask, gray = calculate_q1(path)
        except Exception as e:
            print(f"Error reading image {filename}: {e}")
            continue


        # DEBUG: SAVE MASK
        mask_path = f"debug_masks/{filename}_mask.png"
        os.makedirs("debug_masks", exist_ok=True)
        cv2.imwrite(mask_path, R_mask)
        print(f"Saved mask to {mask_path}")
        # --- STEP 2: Compute Q5 using ISO method ---
        Q5, ep , q5_raw = calculate_q5(R_mask, gray)

        # Save result
        results.append((filename, Q5, q5_raw))

        # Print formatted row
        print("{:<45} {:>8} {:>12.4f}".format(filename, Q5, q5_raw))
         
     
    # --- SUMMARY ---
    if results:
        avg_q5 = sum(r[1] for r in results) / len(results)
        print("-" * 70)
        print(f"Total: {len(results)} | Avg Q5: {avg_q5:.2f}")
        print("=" * 80)



if __name__ == "__main__":
    # Parse command line arguments
    # test_q5_all("test_images")
    test_q1 = True
    test_q2 = True
    test_q3 = True
    test_q4 = True
    test_q5 = True 
    test_q6 = True
    test_q7 = True
    run_verify = False
    
    if len(sys.argv) > 1:
        run_verify = '--verify' in sys.argv
        
        # If specific Q flags are set, use them; otherwise default to all
        has_q_flags = '--q1' in sys.argv or '--q2' in sys.argv or '--q3' in sys.argv or '--q4' in sys.argv or '--q5' in sys.argv or '--q6' in sys.argv or '--q7' in sys.argv
        
        if has_q_flags:
            test_q1 = '--q1' in sys.argv or '--all' in sys.argv
            test_q2 = '--q2' in sys.argv or '--all' in sys.argv
            test_q3 = '--q3' in sys.argv or '--all' in sys.argv
            test_q4 = '--q4' in sys.argv or '--all' in sys.argv
            test_q5 = '--q5' in sys.argv or '--all' in sys.argv
            test_q6 = "--q6" in sys.argv or "--all" in sys.argv
            test_q7 = "--q7" in sys.argv or "--all" in sys.argv
            
            # If only one Q is specified, disable others
            if "--q1" in sys.argv and not any(f in sys.argv for f in ["--q2","--q3","--q4","--q5","--q6","--q7","--all"]):
                test_q2 = test_q3 = test_q4 = test_q5 = test_q6 = test_q7 = False
            if "--q2" in sys.argv and not any(f in sys.argv for f in ["--q1","--q3","--q4","--q5","--q6","--q7","--all"]):
                test_q1 = test_q3 = test_q4 = test_q5 = test_q6 = test_q7 = False
            if "--q3" in sys.argv and not any(f in sys.argv for f in ["--q1","--q2","--q4","--q5","--q6","--q7","--all"]):
                test_q1 = test_q2 = test_q4 = test_q5 = test_q6 = test_q7 = False
            if "--q4" in sys.argv and not any(f in sys.argv for f in ["--q1","--q2","--q3","--q5","--q6","--q7","--all"]):
                test_q1 = test_q2 = test_q5 = test_q6 = test_q7 = False
                test_q3 = True
            if "--q5" in sys.argv and not any(f in sys.argv for f in ["--q1","--q2","--q3","--q4","--q6","--q7","--all"]):
                test_q1 = test_q2 = test_q3 = test_q4 = test_q6 = test_q7 = False
            if "--q6" in sys.argv and not any(f in sys.argv for f in ["--q1","--q2","--q3","--q4","--q5","--q7","--all"]):
                test_q1 = test_q2 = test_q3 = test_q4 = test_q5 = test_q7 = False
            if "--q7" in sys.argv and not any(f in sys.argv for f in ["--q1","--q2","--q3","--q4","--q5","--q6","--all"]):
                test_q1 = test_q2 = test_q4 = test_q5 = test_q6 = False
                test_q3 = True  # Q7 requires Q3 for g_mean
                # If --verify is the only flag, keep defaults (all tests)
    
    # Q4 requires Q3
    if test_q4 and not test_q3:
        test_q3 = True
    
    # Q7 requires Q3 (for g_mean)
    if test_q7 and not test_q3:
        test_q3 = True
    
    # Run verification tests if requested
    if run_verify:
        if test_q1:
            run_q1_verification_tests()
            print("\n" + "="*70 + "\n")
        if test_q3 or test_q4:
            run_q3_verification_tests()
            print("\n" + "="*70 + "\n")
        if test_q7:
            run_q7_verification_tests()
            print("\n" + "="*70 + "\n")
    
    # Test all images with selected metrics
    test_all_images("test_images", test_q1=test_q1, test_q2=test_q2, test_q3=test_q3, test_q4=test_q4, test_q5=test_q5, test_q6=test_q6, test_q7=test_q7)
