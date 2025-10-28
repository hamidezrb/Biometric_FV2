"""
ISO/IEC 29794-9 Q1 Verification Test Harness
Finger Vascular Biometrics - Second Phalanx

This script verifies the Q1 (Effective Area) implementation against
ISO/IEC 29794-9 Clause 5.2.1 requirements through systematic testing.

Mandatory Verification Tests:
- Test A (Cap Check): Input area > 20000 pixels. Must return Q1 = 100
- Test B (Scaling Check): Input area = 10000 pixels. Must return Q1 = 50  
- Test C (Veto Check): Input area = 0 pixels. Must return Q1 = 0
"""

import os
import tempfile
from quality_metrics import calculate_q1, create_test_image
from q2 import calculate_q2
import pandas as pd


def run_verification_tests() -> None:
    """
    Run comprehensive verification tests for Q1 implementation.
    
    Tests the three mandatory verification scenarios:
    1. Cap Check: Area > 20000 pixels → Q1 = 100
    2. Scaling Check: Area = 10000 pixels → Q1 = 50
    3. Veto Check: Area = 0 pixels → Q1 = 0
    """
    
    print("=" * 80)
    print("ISO/IEC 29794-9 Q1 (Effective Area) VERIFICATION TESTS")
    print("Finger Vascular Biometrics - Second Phalanx")
    print("=" * 80)
    print()
    
    # Create directory for test images
    with tempfile.TemporaryDirectory() as temp_dir:
        
        # Test A: Cap Check (> 20000 pixels → Q1 = 100)
        print("TEST A: Cap Check")
        print("-" * 40)
        
        temp_dir = "test_outputs"
        os.makedirs(temp_dir, exist_ok=True)
        
        test_a_path = os.path.join(temp_dir, "test_a_cap.bmp")
        create_test_image(200, 200, 25000, test_a_path)  # > 20000 pixels
        q1_a, pixels_a = calculate_q1(test_a_path)
        expected_a = 100
        status_a = "PASS" if q1_a == expected_a else "FAIL"
        print(f"Input pixels: {pixels_a}")
        print(f"Expected Q1: {expected_a}")
        print(f"Actual Q1: {q1_a}")
        print(f"Status: {status_a}")
        print()
        
        # Test B: Scaling Check (10000 pixels → Q1 = 50)
        print("TEST B: Scaling Check")
        print("-" * 40)
        test_b_path = os.path.join(temp_dir, "test_b_scaling.bmp")
        create_test_image(200, 200, 10000, test_b_path)  # Exactly 10000 pixels
        q1_b, pixels_b = calculate_q1(test_b_path)
        expected_b = 50
        status_b = "PASS" if q1_b == expected_b else "FAIL"
        print(f"Input pixels: {pixels_b}")
        print(f"Expected Q1: {expected_b}")
        print(f"Actual Q1: {q1_b}")
        print(f"Status: {status_b}")
        print()
        
        # Test C: Veto Check (0 pixels → Q1 = 0)
        print("TEST C: Veto Check")
        print("-" * 40)
        test_c_path = os.path.join(temp_dir, "test_c_veto.bmp")
        create_test_image(200, 200, 0, test_c_path)  # 0 pixels
        q1_c, pixels_c = calculate_q1(test_c_path)
        expected_c = 0
        status_c = "PASS" if q1_c == expected_c else "FAIL"
        print(f"Input pixels: {pixels_c}")
        print(f"Expected Q1: {expected_c}")
        print(f"Actual Q1: {q1_c}")
        print(f"Status: {status_c}")
        print()
        
        # Additional Edge Case Tests
        print("ADDITIONAL EDGE CASE TESTS")
        print("-" * 40)
        
        # Test D: Exact threshold (20000 pixels → Q1 = 100)
        print("TEST D: Exact Threshold Check")
        test_d_path = os.path.join(temp_dir, "test_d_exact.bmp")
        create_test_image(200, 200, 20000, test_d_path)  # Exactly 20000 pixels
        q1_d, pixels_d = calculate_q1(test_d_path)
        expected_d = 100
        status_d = "PASS" if q1_d == expected_d else "FAIL"
        print(f"Input pixels: {pixels_d}")
        print(f"Expected Q1: {expected_d}")
        print(f"Actual Q1: {q1_d}")
        print(f"Status: {status_d}")
        print()
        
        # Test E: Half threshold (10000 pixels → Q1 = 50)
        print("TEST E: Half Threshold Check")
        test_e_path = os.path.join(temp_dir, "test_e_half.bmp")
        create_test_image(200, 200, 10000, test_e_path)  # Exactly 10000 pixels
        q1_e, pixels_e = calculate_q1(test_e_path)
        expected_e = 50
        status_e = "PASS" if q1_e == expected_e else "FAIL"
        print(f"Input pixels: {pixels_e}")
        print(f"Expected Q1: {expected_e}")
        print(f"Actual Q1: {q1_e}")
        print(f"Status: {status_e}")
        print()
        
        # Test F: Quarter threshold (5000 pixels → Q1 = 25)
        print("TEST F: Quarter Threshold Check")
        test_f_path = os.path.join(temp_dir, "test_f_quarter.bmp")
        create_test_image(200, 200, 5000, test_f_path)  # 5000 pixels
        q1_f, pixels_f = calculate_q1(test_f_path)
        expected_f = 25
        status_f = "PASS" if q1_f == expected_f else "FAIL"
        print(f"Input pixels: {pixels_f}")
        print(f"Expected Q1: {expected_f}")
        print(f"Actual Q1: {q1_f}")
        print(f"Status: {status_f}")
        print()
        
        # Summary
        print("=" * 80)
        print("VERIFICATION SUMMARY")
        print("=" * 80)
        all_tests = [
            ("Test A (Cap Check)", status_a),
            ("Test B (Scaling Check)", status_b),
            ("Test C (Veto Check)", status_c),
            ("Test D (Exact Threshold)", status_d),
            ("Test E (Half Threshold)", status_e),
            ("Test F (Quarter Threshold)", status_f)
        ]
        
        passed_tests = sum(1 for _, status in all_tests if status == "PASS")
        total_tests = len(all_tests)
        
        for test_name, status in all_tests:
            print(f"{test_name}: {status}")
        
        print()
        print(f"Overall Result: {passed_tests}/{total_tests} tests PASSED")
        
        if passed_tests == total_tests:
            print("✅ ALL TESTS PASSED - ISO/IEC 29794-9 Q1 Implementation is CORRECT")
        else:
            print("❌ SOME TESTS FAILED - ISO/IEC 29794-9 Q1 Implementation needs REVIEW")
        
        print("=" * 80)


def test_with_real_image_Q1(image_path: str) -> None:
    """
    Test Q1 calculation with a real image file.
    
    Args:
        image_path (str): Path to a real image file
    """
    print("=" * 80)
    print("REAL IMAGE TEST")
    print("=" * 80)
    
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found")
        return
    
    q1_score, pixel_count = calculate_q1(image_path)
    
    print(f"Image: {image_path}")
    print(f"Foreground pixels: {pixel_count}")
    print(f"Q1 Score: {q1_score}")
    print(f"Quality Assessment: {'High' if q1_score >= 80 else 'Medium' if q1_score >= 50 else 'Low'}")
    print("=" * 80)


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
    # Run the main verification tests
    # run_verification_tests()
    test_with_real_image_Q1("test_images/test_file_068.png")
    test_with_real_image_Q2("test_images/test_file_068.png")
    # batch_test_with_real_image_Q2()
    
    # Uncomment the line below to test with a real image
    # test_with_real_image_Q1("path/to/your/image.bmp")