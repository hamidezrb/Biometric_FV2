"""ISO/IEC 29794-9 Clause 5.2.1 — foreground, occlusion, and Formula (1) tests."""

import os
import tempfile
import unittest

import cv2
import numpy as np

from iso_constants import CaptureSite, SC_FINGER_SECOND_PHALANX, get_capture_site_coefficients
from iso_foreground import extract_foreground_region, is_foreground_region_valid
from iso_occlusion import compute_unoccluded_foreground, detect_occlusion_mask
from q1 import calculate_q1, calculate_q1_detailed, compute_q1_raw, compute_q1_score


class TestForegroundPolarity(unittest.TestCase):
    def test_dark_background_subject_not_inverted(self):
        """Low-quality finger on black background must not label background as R."""
        gray = np.zeros((192, 736), dtype=np.uint8)
        gray[60:132, 40:700] = 180
        gray[60:132, 650:700] = 255  # saturated tip

        R, S = extract_foreground_region(gray)
        self.assertGreater(S, 0)
        self.assertLess(S, gray.size // 2)
        self.assertGreater(int(np.count_nonzero(R == 255)), 1000)


class TestOcclusionDetection(unittest.TestCase):
    def test_saturated_pixels_excluded_from_unoccluded(self):
        gray = np.zeros((100, 100), dtype=np.uint8)
        gray[20:80, 20:80] = 200
        gray[20:40, 60:80] = 255

        R, S_fg = extract_foreground_region(gray)
        occ = detect_occlusion_mask(gray, R)
        _, _, S_occ, S_unocc = compute_unoccluded_foreground(R, occ)

        self.assertEqual(S_unocc, S_fg - S_occ)
        self.assertGreater(S_occ, 0)
        self.assertLess(S_unocc, S_fg)

    def test_no_occlusion_when_no_saturation(self):
        gray = np.zeros((80, 80), dtype=np.uint8)
        gray[20:60, 20:60] = 200

        R, S_fg = extract_foreground_region(gray)
        occ = detect_occlusion_mask(gray, R)
        _, _, S_occ, S_unocc = compute_unoccluded_foreground(R, occ)

        self.assertEqual(S_occ, 0)
        self.assertEqual(S_unocc, S_fg)


class TestQ1Formula1WithOcclusion(unittest.TestCase):
    def test_formula_uses_unoccluded_not_foreground(self):
        S_fg = 11511
        S_occ = 7665
        S_unocc = S_fg - S_occ
        self.assertEqual(compute_q1_score(S_unocc, SC_FINGER_SECOND_PHALANX), 19)
        self.assertEqual(compute_q1_score(S_fg, SC_FINGER_SECOND_PHALANX), 58)

    def test_q1_raw_before_rounding(self):
        self.assertAlmostEqual(compute_q1_raw(11511, 20000), 57.555)
        self.assertAlmostEqual(compute_q1_raw(3846, 20000), 19.23)

    def test_calculate_q1_detailed_fields(self):
        gray = np.zeros((100, 100), dtype=np.uint8)
        gray[20:80, 20:80] = 200
        gray[20:35, 60:75] = 255

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "q1.png")
            cv2.imwrite(path, gray)
            result = calculate_q1_detailed(path, Sc=1600)

        self.assertLess(result.S_unoccluded, result.S_foreground)
        self.assertEqual(result.S_unoccluded, result.S_foreground - result.S_occluded)
        self.assertEqual(result.Q1_score, compute_q1_score(result.S_unoccluded, 1600))

    def test_invalid_unoccluded_returns_zero(self):
        self.assertFalse(is_foreground_region_valid(0))
        self.assertEqual(compute_q1_score(0, SC_FINGER_SECOND_PHALANX), 0)


class TestQ1Integration(unittest.TestCase):
    def test_low_quality_image_path_if_present(self):
        path = os.path.join(
            "test_images", "low_quality", "PLUS-FV3-Laser_PALMAR_018_01_04_01.png"
        )
        if not os.path.isfile(path):
            self.skipTest("reference image not available")

        result = calculate_q1_detailed(path, capture_site=CaptureSite.FINGER_SECOND_PHALANX)
        self.assertNotEqual(result.S_foreground, result.S_unoccluded)
        self.assertLess(result.Q1_score, compute_q1_score(result.S_foreground, result.Sc))
        self.assertNotEqual(result.Q1_score, 100)


if __name__ == "__main__":
    unittest.main()
