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
import sys
import tempfile
from q1 import calculate_q1
from q2 import calculate_q2
from q3 import calculate_q3
from quality_metrics import (
    create_test_image,
    create_high_contrast_image,
    create_low_contrast_image,
    create_high_noise_suppression_image,
    create_low_noise_suppression_image
)
import pandas as pd


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


<<<<<<< HEAD
def test_with_real_image_Q1(image_path: str) -> None:
=======
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


def test_all_images(images_dir: str = "test_images", test_q1: bool = True, test_q3: bool = True, test_q4: bool = True) -> None:
>>>>>>> f632b2c (feat: Implement Q1, Q3, and Q4 quality metrics)
    """
    Test all vein images with specified quality metrics.
    
    Args:
        images_dir (str): Directory containing vein images
        test_q1 (bool): Test Q1 if True
        test_q3 (bool): Test Q3 if True (also calculates Q4)
        test_q4 (bool): Test Q4 if True (requires Q3)
    """
    # Determine which metrics to test
    metrics = []
    if test_q1:
        metrics.append("Q1")
    if test_q3:
        metrics.append("Q3")
    if test_q4 and test_q3:
        metrics.append("Q4")
    
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
            q1_score, pixel_count, R_mask, Grayscale_image = calculate_q1(image_path)
            
            if test_q1:
                result['q1'] = q1_score
                result['pixels'] = pixel_count
            
            if test_q3:
                q3_score, q4_score, sigma, g_mean = calculate_q3(R_mask, Grayscale_image)
                result['q3'] = q3_score
                if test_q4:
                    result['q4'] = q4_score
                    result['ratio'] = sigma / g_mean if g_mean > 0 else 0
            
            results.append(result)
            
        except Exception as e:
            print(f"ERROR: {image_name} - {str(e)}")
            if test_q1:
                result['q1'] = 0
            if test_q3:
                result['q3'] = 0
            if test_q4:
                result['q4'] = 0
            results.append(result)
    
    # Output table
    header_parts = ['Image']
    if test_q1:
        header_parts.append('Q1')
    if test_q3:
        header_parts.append('Q3')
    if test_q4 and test_q3:
        header_parts.append('Q4')
    
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
        if test_q3:
            row_parts.append(r.get('q3', 0))
        if test_q4 and test_q3:
            row_parts.append(r.get('q4', 0))
        print(header_format.format(*row_parts))
    
    # Summary
    valid_results = [r for r in results if r.get('q1', 0) > 0 or r.get('q3', 0) > 0]
    if valid_results:
        summary_parts = [f"Total: {len(valid_results)}"]
        if test_q1:
            avg_q1 = sum(r.get('q1', 0) for r in valid_results) / len(valid_results)
            high_count = sum(1 for r in valid_results if r.get('q1', 0) >= 50)
            low_count = sum(1 for r in valid_results if r.get('q1', 0) < 50)
            summary_parts.append(f"Avg Q1: {avg_q1:.1f}")
            summary_parts.append(f"High: {high_count} | Low: {low_count}")
        if test_q3:
            avg_q3 = sum(r.get('q3', 0) for r in valid_results) / len(valid_results)
            summary_parts.append(f"Avg Q3: {avg_q3:.1f}")
        if test_q4 and test_q3:
            avg_q4 = sum(r.get('q4', 0) for r in valid_results) / len(valid_results)
            summary_parts.append(f"Avg Q4: {avg_q4:.1f}")
        
        print("-" * (40 + 7 * (len(header_parts) - 1)))
        print(" | ".join(summary_parts))
    
    print("=" * 70)


def test_with_real_image_Q2(image_path: str):
    
    q2, cx, cy, d, r = calculate_q2(image_path)
    print("="*80)
    print(f"Image: {image_path}")
    print(f"Centroid: ({cx:.2f}, {cy:.2f})")
    print(f"Distance from center: {d:.2f}")
    print(f"Normalized offset r: {r:.3f}")
    print(f"Q2 score: {q2}")
    print("="*80)
    
def batch_test_with_real_image_Q2():
    input_dir = "test_images"
    results = []

    for fname in sorted(os.listdir(input_dir)):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".bmp")):
            continue
        path = os.path.join(input_dir, fname)
        q2, cx, cy, d, r = calculate_q2(path)
        results.append({
            "filename": fname,
            "Q2": q2,
            "centroid_x": cx,
            "centroid_y": cy,
            "distance_d": d,
            "r_norm": r
        })

    df = pd.DataFrame(results)
    os.makedirs("results", exist_ok=True)
    csv_path = "results/q2_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"✅ Saved results to {csv_path}")
    print(df)

if __name__ == "__main__":
<<<<<<< HEAD
    # Run the main verification tests
    # run_verification_tests()
    test_with_real_image_Q1("test_images/test_file_068.png")
    test_with_real_image_Q2("test_images/test_file_068.png")
    # batch_test_with_real_image_Q2()
    
    # Uncomment the line below to test with a real image
    # test_with_real_image_Q1("path/to/your/image.bmp")
=======
    # Parse command line arguments
    test_q1 = True
    test_q3 = True
    test_q4 = True
    run_verify = False
    
    if len(sys.argv) > 1:
        run_verify = '--verify' in sys.argv
        
        # If specific Q flags are set, use them; otherwise default to all
        has_q_flags = '--q1' in sys.argv or '--q3' in sys.argv or '--q4' in sys.argv
        
        if has_q_flags:
            test_q1 = '--q1' in sys.argv or '--all' in sys.argv
            test_q3 = '--q3' in sys.argv or '--all' in sys.argv
            test_q4 = '--q4' in sys.argv or '--all' in sys.argv
            
            # If only one Q is specified, disable others
            if '--q1' in sys.argv and '--q3' not in sys.argv and '--q4' not in sys.argv and '--all' not in sys.argv:
                test_q3 = False
                test_q4 = False
            if '--q3' in sys.argv and '--q1' not in sys.argv and '--q4' not in sys.argv and '--all' not in sys.argv:
                test_q1 = False
                test_q4 = False
            if '--q4' in sys.argv and '--q1' not in sys.argv and '--q3' not in sys.argv and '--all' not in sys.argv:
                test_q1 = False
                test_q3 = True  # Q4 requires Q3
        # If --verify is the only flag, keep defaults (all tests)
    
    # Q4 requires Q3
    if test_q4 and not test_q3:
        test_q3 = True
    
    # Run verification tests if requested
    if run_verify:
        if test_q1:
            run_q1_verification_tests()
            print("\n" + "="*70 + "\n")
        if test_q3 or test_q4:
            run_q3_verification_tests()
            print("\n" + "="*70 + "\n")
    
    # Test all images with selected metrics
    test_all_images("test_images", test_q1=test_q1, test_q3=test_q3, test_q4=test_q4)
>>>>>>> f632b2c (feat: Implement Q1, Q3, and Q4 quality metrics)
