"""Tests for ISO/IEC 29794-9 capture-site coefficient selection (Tables 1–3)."""

import inspect
import unittest

from iso_constants import (
    CAPTURE_SITE_COEFFICIENTS,
    CaptureSite,
    FC_FINGER_SECOND_PHALANX,
    LC_FINGER_SECOND_PHALANX,
    SC_FINGER_SECOND_PHALANX,
    get_capture_site_coefficients,
)
from q1 import calculate_q1
from q8 import calculate_q8
from q9 import calculate_q9


class TestCaptureSiteCoefficients(unittest.TestCase):
    def test_finger_coefficients(self):
        c = get_capture_site_coefficients(CaptureSite.FINGER_SECOND_PHALANX)
        self.assertEqual(c["SC"], 20000)
        self.assertEqual(c["LC"], 600)
        self.assertEqual(c["FC"], 15)

    def test_palm_or_dorsal_coefficients(self):
        c = get_capture_site_coefficients(CaptureSite.PALM_OR_DORSAL)
        self.assertEqual(c["SC"], 40000)
        self.assertEqual(c["LC"], 1200)
        self.assertEqual(c["FC"], 25)

    def test_full_hand_coefficients(self):
        c = get_capture_site_coefficients(CaptureSite.FULL_HAND)
        self.assertEqual(c["SC"], 300000)
        self.assertEqual(c["LC"], 9000)
        self.assertEqual(c["FC"], 50)

    def test_backward_compatible_aliases(self):
        finger = CAPTURE_SITE_COEFFICIENTS[CaptureSite.FINGER_SECOND_PHALANX]
        self.assertEqual(SC_FINGER_SECOND_PHALANX, finger["SC"])
        self.assertEqual(LC_FINGER_SECOND_PHALANX, finger["LC"])
        self.assertEqual(FC_FINGER_SECOND_PHALANX, finger["FC"])

    def test_default_capture_site_is_finger(self):
        for fn in (calculate_q1, calculate_q8, calculate_q9):
            sig = inspect.signature(fn)
            self.assertEqual(
                sig.parameters["capture_site"].default,
                CaptureSite.FINGER_SECOND_PHALANX,
            )

    def test_q1_sc_differs_by_capture_site(self):
        S = 10000
        sc_finger = get_capture_site_coefficients(CaptureSite.FINGER_SECOND_PHALANX)["SC"]
        sc_palm = get_capture_site_coefficients(CaptureSite.PALM_OR_DORSAL)["SC"]
        sc_full = get_capture_site_coefficients(CaptureSite.FULL_HAND)["SC"]
        q_finger = min(100, int(round(S / sc_finger * 100)))
        q_palm = min(100, int(round(S / sc_palm * 100)))
        q_full = min(100, int(round(S / sc_full * 100)))
        self.assertEqual(q_finger, 50)
        self.assertEqual(q_palm, 25)
        self.assertEqual(q_full, 3)

    def test_q8_lc_differs_by_capture_site(self):
        Nv = 600
        lc_finger = get_capture_site_coefficients(CaptureSite.FINGER_SECOND_PHALANX)["LC"]
        lc_palm = get_capture_site_coefficients(CaptureSite.PALM_OR_DORSAL)["LC"]
        lc_full = get_capture_site_coefficients(CaptureSite.FULL_HAND)["LC"]
        self.assertEqual(min(100, int(round(Nv / lc_finger * 100))), 100)
        self.assertEqual(min(100, int(round(Nv / lc_palm * 100))), 50)
        self.assertEqual(min(100, int(round(Nv / lc_full * 100))), 7)

    def test_q9_fc_differs_by_capture_site(self):
        N_fp = 15
        fc_finger = get_capture_site_coefficients(CaptureSite.FINGER_SECOND_PHALANX)["FC"]
        fc_palm = get_capture_site_coefficients(CaptureSite.PALM_OR_DORSAL)["FC"]
        fc_full = get_capture_site_coefficients(CaptureSite.FULL_HAND)["FC"]
        self.assertEqual(min(100, int(round(N_fp / fc_finger * 100))), 100)
        self.assertEqual(min(100, int(round(N_fp / fc_palm * 100))), 60)
        self.assertEqual(min(100, int(round(N_fp / fc_full * 100))), 30)


if __name__ == "__main__":
    unittest.main()
