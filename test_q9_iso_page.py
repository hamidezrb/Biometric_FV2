"""ISO/IEC 29794-9 Clause 5.2.9 (Q9 feature points page) compliance tests."""



import unittest



import numpy as np



from iso_constants import CaptureSite, get_capture_site_coefficients

from q9 import (

    calculate_q9,

    compute_q9_score,

    count_feature_points_in_foreground,

    transition_count_8nbr,

)

from vessel_utils import prepare_vessel_skeleton, zs_thinning





class TestQ9Formula16(unittest.TestCase):

    def test_formula_16_exact(self):

        self.assertEqual(compute_q9_score(15, 15), 100)



    def test_formula_16_half(self):

        self.assertEqual(compute_q9_score(8, 16), 50)



    def test_formula_16_clamps_above_100(self):

        self.assertEqual(compute_q9_score(500, 15), 100)



    def test_fc_zero_returns_zero(self):

        self.assertEqual(compute_q9_score(10, 0), 0)





class TestQ9Table3Fc(unittest.TestCase):

    def test_finger_fc_15(self):

        self.assertEqual(

            get_capture_site_coefficients(CaptureSite.FINGER_SECOND_PHALANX)["FC"],

            15,

        )



    def test_palm_fc_25(self):

        self.assertEqual(

            get_capture_site_coefficients(CaptureSite.PALM_OR_DORSAL)["FC"],

            25,

        )



    def test_full_hand_fc_50(self):

        self.assertEqual(

            get_capture_site_coefficients(CaptureSite.FULL_HAND)["FC"],

            50,

        )





class TestQ9TransitionRules(unittest.TestCase):

    def test_cross_center_transition_count_8(self):

        skel = np.zeros((5, 5), dtype=np.uint8)

        skel[2, 1:4] = 1

        skel[1:4, 2] = 1

        self.assertEqual(transition_count_8nbr(skel, 2, 2), 8)



    def test_line_endpoint_transition_count_2(self):

        skel = np.zeros((5, 7), dtype=np.uint8)

        skel[2, 1:6] = 1

        self.assertEqual(transition_count_8nbr(skel, 2, 1), 2)

        self.assertEqual(transition_count_8nbr(skel, 2, 5), 2)



    def test_isolated_pixel_not_endpoint(self):

        skel = np.zeros((5, 5), dtype=np.uint8)

        skel[2, 2] = 1

        self.assertEqual(transition_count_8nbr(skel, 2, 2), 0)



    def test_t_junction_transition_count_6_or_8(self):

        skel = np.zeros((7, 7), dtype=np.uint8)

        skel[1:5, 3] = 1

        skel[3, 1:6] = 1

        t = transition_count_8nbr(skel, 3, 3)

        self.assertIn(t, (6, 8))





class TestQ9FeatureCounting(unittest.TestCase):

    def _fg_full(self, shape):

        return np.ones(shape, dtype=bool)



    def test_one_pixel_line_two_endpoints(self):

        skel = np.zeros((11, 21), dtype=np.uint8)

        skel[5, 5:16] = 1

        fg = self._fg_full(skel.shape)

        n_end, n_int, n_fp = count_feature_points_in_foreground(skel, fg)

        self.assertEqual(n_end, 2)

        self.assertEqual(n_int, 0)

        self.assertEqual(n_fp, 2)



    def test_cross_one_intersection_four_endpoints(self):

        skel = np.zeros((7, 7), dtype=np.uint8)

        skel[3, 1:6] = 1

        skel[1:6, 3] = 1

        fg = self._fg_full(skel.shape)

        n_end, n_int, n_fp = count_feature_points_in_foreground(skel, fg)

        self.assertEqual(n_int, 1)

        self.assertEqual(n_end, 4)

        self.assertEqual(n_fp, 5)



    def test_border_endpoint_counted(self):

        skel = np.zeros((5, 11), dtype=np.uint8)

        skel[2, 0:8] = 1

        fg = self._fg_full(skel.shape)

        n_end, _, _ = count_feature_points_in_foreground(skel, fg)

        self.assertEqual(n_end, 2)

        self.assertEqual(transition_count_8nbr(skel, 2, 0), 2)



    def test_vessel_outside_foreground_not_counted(self):

        R = np.zeros((30, 30), dtype=np.uint8)

        R[5:25, 5:25] = 255

        vein = np.zeros((30, 30), dtype=np.uint8)

        vein[15, :] = 255

        fg = R == 255

        _, skel = prepare_vessel_skeleton(R, vein, iso_minimal=True)

        n_end, n_int, n_fp = count_feature_points_in_foreground(skel, fg)

        Q9, n_fp2, _, _, _ = calculate_q9(R, vein)

        self.assertEqual(n_fp, n_fp2)

        self.assertEqual(n_end, 2)

        self.assertEqual(n_int, 0)





class TestQ9EdgeCases(unittest.TestCase):

    def test_empty_foreground_mask(self):

        R = np.zeros((16, 16), dtype=np.uint8)

        vein = np.zeros((16, 16), dtype=np.uint8)

        Q9, n_fp, n_end, n_int, skel = calculate_q9(R, vein, S_unoccluded=0)

        self.assertEqual(Q9, 0)

        self.assertEqual(n_fp, 0)

        self.assertEqual(n_end, 0)

        self.assertEqual(n_int, 0)



    def test_foreground_no_vessel_pixels(self):

        R = np.zeros((12, 12), dtype=np.uint8)

        R[2:10, 2:10] = 255

        vein = np.zeros((12, 12), dtype=np.uint8)

        Q9, n_fp, _, _, _ = calculate_q9(R, vein)

        self.assertEqual(n_fp, 0)

        self.assertEqual(Q9, 0)



    def test_vein_mask_255_binarized_to_01(self):

        R = np.zeros((20, 20), dtype=np.uint8)

        R[:, :] = 255

        vein = np.zeros((20, 20), dtype=np.uint8)

        vein[10, 4:16] = 255

        _, skel = prepare_vessel_skeleton(R, vein, iso_minimal=True)

        self.assertTrue(np.all(np.isin(skel, (0, 1))))



    def test_thick_vessel_thinned_before_counting(self):

        R = np.zeros((30, 30), dtype=np.uint8)

        R[:, :] = 255

        vein = np.zeros((30, 30), dtype=np.uint8)

        vein[12:17, 8:22] = 255

        Q9a, n_fp, n_end, _, _ = calculate_q9(R, vein)

        vein_thin = np.zeros_like(vein)

        vein_thin[14, 8:22] = 255

        _, skel_thin = prepare_vessel_skeleton(R, vein_thin)

        n_end_thin, _, n_fp_thin = count_feature_points_in_foreground(

            skel_thin, R == 255

        )

        self.assertEqual(n_end_thin, 2)

        self.assertGreater(n_fp, 0)



    def test_capture_site_fc_in_calculate_q9(self):

        R = np.zeros((15, 15), dtype=np.uint8)

        R[:, :] = 255

        vein = np.zeros((15, 15), dtype=np.uint8)

        vein[7, 3:12] = 255

        Q_finger, n_fp, _, _, _ = calculate_q9(

            R, vein, capture_site=CaptureSite.FINGER_SECOND_PHALANX

        )

        Q_palm, _, _, _, _ = calculate_q9(

            R, vein, capture_site=CaptureSite.PALM_OR_DORSAL

        )

        self.assertEqual(Q_finger, compute_q9_score(n_fp, 15))

        self.assertEqual(Q_palm, compute_q9_score(n_fp, 25))

        self.assertLessEqual(Q_palm, Q_finger)





if __name__ == "__main__":

    unittest.main()


