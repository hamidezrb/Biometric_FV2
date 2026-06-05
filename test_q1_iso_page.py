"""ISO/IEC 29794-9 Clause 5.2.1 (Q1 page) — Formula (1) and Table 1 compliance tests."""

import os
import tempfile
import unittest

import cv2
import numpy as np

from iso_constants import CaptureSite, get_capture_site_coefficients
from iso_foreground import is_foreground_region_valid
from q1 import calculate_q1, compute_q1_score


class TestQ1Formula1(unittest.TestCase):
    def test_formula_1_exact(self):
        self.assertEqual(compute_q1_score(10000, 20000), 50)
        self.assertEqual(compute_q1_score(20000, 20000), 100)
        self.assertEqual(compute_q1_score(25000, 20000), 100)

    def test_invalid_foreground_zero_pixels(self):
        self.assertFalse(is_foreground_region_valid(0))
        self.assertEqual(compute_q1_score(0, 20000), 0)

    def test_table_1_sc_by_capture_site(self):
        self.assertEqual(
            get_capture_site_coefficients(CaptureSite.FINGER_SECOND_PHALANX)["SC"],
            20000,
        )
        self.assertEqual(
            get_capture_site_coefficients(CaptureSite.PALM_OR_DORSAL)["SC"],
            40000,
        )
        self.assertEqual(
            get_capture_site_coefficients(CaptureSite.FULL_HAND)["SC"],
            300000,
        )

    def test_calculate_q1_uses_table_1_default_finger(self):
        gray = np.zeros((80, 80), dtype=np.uint8)
        gray[20:60, 20:60] = 200
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "fg.png")
            cv2.imwrite(path, gray)
            Q1, S, _, _ = calculate_q1(path)
        expected = compute_q1_score(
            S, get_capture_site_coefficients(CaptureSite.FINGER_SECOND_PHALANX)["SC"]
        )
        self.assertEqual(Q1, expected)

    def test_unreadable_image_returns_q1_zero(self):
        Q1, S, R, gray = calculate_q1(os.path.join("nonexistent", "missing.png"))
        self.assertEqual(Q1, 0)
        self.assertEqual(S, 0)


if __name__ == "__main__":
    unittest.main()
