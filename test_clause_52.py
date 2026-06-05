"""Unit tests for ISO/IEC 29794-9 Clause 5.2 quality components."""

import os
import tempfile
import unittest

import cv2
import numpy as np

from iso_constants import LC_FINGER_SECOND_PHALANX, SC_FINGER_SECOND_PHALANX
from iso_foreground import extract_foreground_region
from q1 import calculate_q1, compute_q1_score
from q2 import calculate_q2
from q5 import calculate_q5
from q6 import calculate_q6
from q7 import calculate_q7
from q8 import calculate_q8
from q9 import calculate_q9
from vessel_utils import zs_thinning


class TestQ1EffectiveArea(unittest.TestCase):
    def test_invalid_foreground_returns_zero(self):
        """Q1 veto when S_unoccluded = 0 (invalid R)."""
        gray = np.zeros((50, 50), dtype=np.uint8)
        R = np.zeros_like(gray)
        S = int(np.count_nonzero(R == 255))
        self.assertEqual(S, 0)
        self.assertEqual(compute_q1_score(S, SC_FINGER_SECOND_PHALANX), 0)

    def test_known_area_maps_to_score(self):
        gray = np.zeros((120, 120), dtype=np.uint8)
        gray[10:90, 10:90] = 220
        R, S = extract_foreground_region(gray)
        self.assertGreater(S, 0)
        Sc = S
        expected = min(100, int(round(S / float(Sc) * 100.0)))
        self.assertEqual(expected, 100)

    def test_calculate_q1_from_file(self):
        gray = np.zeros((80, 80), dtype=np.uint8)
        gray[20:60, 20:60] = 200
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "fg.png")
            cv2.imwrite(path, gray)
            Q1, S, R, _ = calculate_q1(path, Sc=1600)
        self.assertEqual(Q1, min(100, int(round(S / 1600.0 * 100))))
        self.assertGreater(S, 0)


class TestQ2OffsetComplement(unittest.TestCase):
    def test_invalid_r_returns_zero(self):
        R = np.zeros((100, 100), dtype=np.uint8)
        gray = np.zeros((100, 100), dtype=np.uint8)
        Q2, cx, cy, sh, sv = calculate_q2(R, gray)
        self.assertEqual(Q2, 0)
        self.assertIsNone(cx)

    def test_centered_mask_high_score(self):
        gray = np.zeros((200, 200), dtype=np.uint8)
        R = np.zeros_like(gray)
        R[70:130, 70:130] = 255
        gray[R == 255] = 180
        Q2, _, _, _, _ = calculate_q2(R, gray)
        self.assertGreaterEqual(Q2, 90)
        self.assertLessEqual(Q2, 100)


class TestQ6Sharpness(unittest.TestCase):
    def test_invalid_foreground_returns_zero(self):
        R = np.zeros((64, 64), dtype=np.uint8)
        gray = np.zeros((64, 64), dtype=np.uint8)
        Q6, N100 = calculate_q6(R, gray, S_unoccluded=0)
        self.assertEqual(Q6, 0)
        self.assertEqual(N100, 0)

    def test_checkerboard_produces_positive_score(self):
        H, W = 64, 64
        gray = np.zeros((H, W), dtype=np.uint8)
        R = np.zeros_like(gray)
        R[8:56, 8:56] = 255
        block = 4
        for y in range(8, 56, block):
            for x in range(8, 56, block):
                if ((x // block) + (y // block)) % 2 == 0:
                    gray[y:y + block, x:x + block] = 255
                else:
                    gray[y:y + block, x:x + block] = 0
        S = int(np.count_nonzero(R == 255))
        Q6a, N100a = calculate_q6(R, gray, S_unoccluded=S)
        Q6b, N100b = calculate_q6(R, gray, S_unoccluded=S)
        self.assertEqual(Q6a, Q6b)
        self.assertGreater(N100a, 0)
        self.assertGreater(Q6a, 0)
        self.assertLessEqual(Q6a, 100)
        self.assertGreaterEqual(Q6a, 0)


class TestQ7BrightnessUniformity(unittest.TestCase):
    def test_uniform_foreground_near_max(self):
        gray = np.full((50, 50), 128, dtype=np.uint8)
        R = np.zeros_like(gray)
        R[5:45, 5:45] = 255
        Q7, var = calculate_q7(R, gray)
        self.assertEqual(var, 0.0)
        self.assertGreaterEqual(Q7, 99)
        self.assertLessEqual(Q7, 100)

    def test_invalid_foreground_returns_zero(self):
        R = np.zeros((40, 40), dtype=np.uint8)
        gray = np.zeros((40, 40), dtype=np.uint8)
        Q7, var = calculate_q7(R, gray)
        self.assertEqual(Q7, 0)
        self.assertEqual(var, 0.0)


class TestQ8VascularLength(unittest.TestCase):
    def test_known_skeleton_pixel_count(self):
        R = np.zeros((40, 40), dtype=np.uint8)
        R[5:35, 5:35] = 255
        vein = np.zeros((40, 40), dtype=np.uint8)
        vein[20, 10:30] = 255
        vessel01 = ((vein > 0) & (R == 255)).astype(np.uint8)
        skel = zs_thinning(vessel01)
        Nv = int(np.count_nonzero(skel))
        Lc = Nv
        Q8, N_vessel, _ = calculate_q8(R, vein, Lc=Lc)
        self.assertEqual(N_vessel, Nv)
        self.assertEqual(Q8, 100)

    def test_invalid_foreground_returns_zero(self):
        R = np.zeros((20, 20), dtype=np.uint8)
        vein = np.zeros((20, 20), dtype=np.uint8)
        Q8, Nv, _ = calculate_q8(R, vein)
        self.assertEqual(Q8, 0)
        self.assertEqual(Nv, 0)


class TestQ9FeaturePoints(unittest.TestCase):
    def test_line_endpoints(self):
        R = np.zeros((30, 30), dtype=np.uint8)
        R[:, :] = 255
        vein = np.zeros((30, 30), dtype=np.uint8)
        vein[15, 5:25] = 255
        Q9, N_fp, N_end, N_int, _ = calculate_q9(R, vein, Fc=15)
        self.assertGreaterEqual(N_end, 2)
        self.assertEqual(N_fp, N_end + N_int)
        self.assertGreaterEqual(Q9, 0)
        self.assertLessEqual(Q9, 100)

    def test_invalid_foreground_returns_zero(self):
        R = np.zeros((20, 20), dtype=np.uint8)
        vein = np.zeros((20, 20), dtype=np.uint8)
        Q9, N_fp, N_end, N_int, _ = calculate_q9(R, vein)
        self.assertEqual(Q9, 0)
        self.assertEqual(N_fp, 0)


class TestQ5Entropy(unittest.TestCase):
    def test_invalid_foreground_returns_zero(self):
        R = np.zeros((32, 32), dtype=np.uint8)
        gray = np.zeros((32, 32), dtype=np.uint8)
        Q5, H = calculate_q5(R, gray)
        self.assertEqual(Q5, 0)
        self.assertEqual(H, 0.0)


if __name__ == "__main__":
    unittest.main()
