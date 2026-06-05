"""ISO/IEC 29794-9 Clause 5.2.6 (Q6 sharpness page) compliance tests."""

import unittest

import cv2
import numpy as np

from q6 import (
    ISO_SOBEL_KERNELS,
    N100_THRESHOLD,
    _convolve_iso_sobel_operators,
    _minmax_normalize_to_uint8,
    calculate_q6,
    compute_q6_score,
)


class TestQ6IsoKernels(unittest.TestCase):
    def test_four_kernels_match_figure_1(self):
        self.assertEqual(len(ISO_SOBEL_KERNELS), 4)
        self.assertEqual(ISO_SOBEL_KERNELS[0].tolist(),
                         [[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        self.assertEqual(ISO_SOBEL_KERNELS[1].tolist(),
                         [[2, 1, 0], [1, 0, -1], [0, -1, -2]])
        self.assertEqual(ISO_SOBEL_KERNELS[2].tolist(),
                         [[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        self.assertEqual(ISO_SOBEL_KERNELS[3].tolist(),
                         [[0, -1, -2], [1, 0, -1], [2, 1, 0]])

    def test_convolution_same_size_as_input(self):
        gray = np.random.randint(0, 255, (48, 64), dtype=np.uint8).astype(np.float32)
        w = _convolve_iso_sobel_operators(gray)
        self.assertEqual(w.shape, gray.shape)


class TestQ6Formula(unittest.TestCase):
    def test_formula_12_with_scale_factor(self):
        # N100=120, gc=0.006, S=20000 -> 120/(0.006*20000)*100 = 100
        self.assertEqual(compute_q6_score(120, 20000, gc=0.006), 100)

    def test_formula_12_clamps_above_100(self):
        self.assertEqual(compute_q6_score(500, 1000, gc=0.006), 100)

    def test_invalid_foreground_returns_zero(self):
        self.assertEqual(compute_q6_score(100, 0), 0)

    def test_gc_zero_returns_zero(self):
        self.assertEqual(compute_q6_score(100, 20000, gc=0), 0)


class TestQ6EdgeCases(unittest.TestCase):
    def test_empty_foreground_mask(self):
        R = np.zeros((32, 32), dtype=np.uint8)
        gray = np.zeros((32, 32), dtype=np.uint8)
        Q6, N100 = calculate_q6(R, gray, S_unoccluded=0)
        self.assertEqual(Q6, 0)
        self.assertEqual(N100, 0)

    def test_all_zero_image(self):
        R = np.zeros((16, 16), dtype=np.uint8)
        R[4:12, 4:12] = 255
        gray = np.zeros((16, 16), dtype=np.uint8)
        Q6, N100 = calculate_q6(R, gray, S_unoccluded=64)
        self.assertEqual(N100, 0)
        self.assertEqual(Q6, 0)

    def test_minmax_constant_response(self):
        w = np.full((10, 10), 5.0, dtype=np.float32)
        out = _minmax_normalize_to_uint8(w)
        self.assertTrue(np.all(out == 0))

    def test_n100_threshold_is_100(self):
        self.assertEqual(N100_THRESHOLD, 100)


class TestQ6CheckerboardDeterministic(unittest.TestCase):
    def test_checkerboard_positive_q6(self):
        H, W = 64, 64
        gray = np.zeros((H, W), dtype=np.uint8)
        R = np.zeros_like(gray)
        R[8:56, 8:56] = 255
        block = 4
        for y in range(8, 56, block):
            for x in range(8, 56, block):
                if ((x // block) + (y // block)) % 2 == 0:
                    gray[y:y + block, x:x + block] = 255
        S = int(np.count_nonzero(R == 255))
        Q6a, _ = calculate_q6(R, gray, S_unoccluded=S)
        Q6b, _ = calculate_q6(R, gray, S_unoccluded=S)
        self.assertEqual(Q6a, Q6b)
        self.assertGreater(Q6a, 0)


if __name__ == "__main__":
    unittest.main()
