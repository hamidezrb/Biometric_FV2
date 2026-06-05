"""ISO/IEC 29794-9 Clause 5.2.8 (Q8 total vascular length page) compliance tests."""



import unittest



import numpy as np



from iso_constants import CaptureSite, get_capture_site_coefficients

from q8 import (

    calculate_q8,

    compute_q8_score,

    count_vessel_pixels_in_foreground,

)

from vessel_utils import prepare_vessel_skeleton, zs_thinning





def _no_2x2_skeleton_blocks(skel01: np.ndarray) -> bool:

    """True if skeleton has no 2×2 block of ones (1-pixel-wide heuristic)."""

    h, w = skel01.shape

    if h < 2 or w < 2:

        return True

    for y in range(h - 1):

        for x in range(w - 1):

            if np.sum(skel01[y : y + 2, x : x + 2]) >= 4:

                return False

    return True





class TestQ8Formula15(unittest.TestCase):

    def test_formula_15_exact(self):

        self.assertEqual(compute_q8_score(600, 600), 100)



    def test_formula_15_half(self):

        self.assertEqual(compute_q8_score(300, 600), 50)



    def test_formula_15_clamps_above_100(self):

        self.assertEqual(compute_q8_score(10000, 600), 100)



    def test_lc_zero_returns_zero(self):

        self.assertEqual(compute_q8_score(100, 0), 0)





class TestQ8Table2Lc(unittest.TestCase):

    def test_finger_lc_600(self):

        self.assertEqual(

            get_capture_site_coefficients(CaptureSite.FINGER_SECOND_PHALANX)["LC"],

            600,

        )



    def test_palm_lc_1200(self):

        self.assertEqual(

            get_capture_site_coefficients(CaptureSite.PALM_OR_DORSAL)["LC"],

            1200,

        )



    def test_full_hand_lc_9000(self):

        self.assertEqual(

            get_capture_site_coefficients(CaptureSite.FULL_HAND)["LC"],

            9000,

        )





class TestQ8EdgeCases(unittest.TestCase):

    def test_empty_foreground_mask(self):

        R = np.zeros((24, 24), dtype=np.uint8)

        vein = np.zeros((24, 24), dtype=np.uint8)

        Q8, Nv, skel = calculate_q8(R, vein, S_unoccluded=0)

        self.assertEqual(Q8, 0)

        self.assertEqual(Nv, 0)

        self.assertEqual(int(np.sum(skel)), 0)



    def test_foreground_no_vessel_pixels(self):

        R = np.zeros((20, 20), dtype=np.uint8)

        R[4:16, 4:16] = 255

        vein = np.zeros((20, 20), dtype=np.uint8)

        Q8, Nv, _ = calculate_q8(R, vein)

        self.assertEqual(Nv, 0)

        self.assertEqual(Q8, 0)



    def test_vessel_outside_foreground_not_counted(self):

        R = np.zeros((40, 40), dtype=np.uint8)

        R[10:30, 10:30] = 255

        vein = np.zeros((40, 40), dtype=np.uint8)

        vein[20, :] = 255

        fg = R == 255

        _, skel = prepare_vessel_skeleton(R, vein, iso_minimal=True)

        Nv = count_vessel_pixels_in_foreground(skel, fg)

        Q8, Nv_calc, _ = calculate_q8(R, vein)

        self.assertEqual(Nv, Nv_calc)

        self.assertLess(Nv, 40)

        self.assertGreater(Nv, 0)



    def test_vein_mask_255_binarized_to_01(self):

        R = np.zeros((32, 32), dtype=np.uint8)

        R[6:26, 6:26] = 255

        vein = np.zeros((32, 32), dtype=np.uint8)

        vein[16, 8:24] = 255

        vessel01, skel = prepare_vessel_skeleton(R, vein, iso_minimal=True)

        self.assertTrue(np.all(np.isin(vessel01, (0, 1))))

        self.assertTrue(np.all(np.isin(skel, (0, 1))))



    def test_already_thin_one_pixel_line(self):

        R = np.zeros((30, 30), dtype=np.uint8)

        R[:, :] = 255

        vein = np.zeros((30, 30), dtype=np.uint8)

        vein[15, 5:25] = 1

        _, skel = prepare_vessel_skeleton(R, vein, iso_minimal=True)

        self.assertEqual(int(np.count_nonzero(skel)), 20)



    def test_thick_vessel_thinned_to_skeleton(self):

        R = np.zeros((40, 40), dtype=np.uint8)

        R[:, :] = 255

        vein = np.zeros((40, 40), dtype=np.uint8)

        vein[18:23, 10:30] = 255

        raw_count = int(np.count_nonzero(vein))

        _, skel = prepare_vessel_skeleton(R, vein, iso_minimal=True)

        Nv = int(np.count_nonzero(skel))

        self.assertGreater(raw_count, Nv)
        self.assertLess(Nv, raw_count)
        self.assertGreaterEqual(Nv, 14)
        self.assertTrue(_no_2x2_skeleton_blocks(skel))



    def test_capture_site_lc_applied_in_calculate_q8(self):

        R = np.zeros((20, 20), dtype=np.uint8)

        R[:, :] = 255

        vein = np.zeros((20, 20), dtype=np.uint8)

        vein[10, 2:18] = 255

        _, skel = prepare_vessel_skeleton(R, vein)

        Nv = int(np.count_nonzero(skel))

        q_finger, _, _ = calculate_q8(

            R, vein, capture_site=CaptureSite.FINGER_SECOND_PHALANX

        )

        q_palm, _, _ = calculate_q8(

            R, vein, capture_site=CaptureSite.PALM_OR_DORSAL

        )

        self.assertEqual(q_finger, compute_q8_score(Nv, 600))

        self.assertEqual(q_palm, compute_q8_score(Nv, 1200))

        self.assertLessEqual(q_palm, q_finger)





if __name__ == "__main__":

    unittest.main()


