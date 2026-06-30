"""OpenVein Q8/Q9 vessel cleanup — ISO-minimal vs heuristic presets."""

import unittest

import numpy as np

from vessel_utils import (
    OpenVeinVesselCleanupConfig,
    binarize_openvein_vein_map,
    prepare_vessel_skeleton,
    zs_thinning,
)


class TestOpenVeinVesselCleanup(unittest.TestCase):
    def test_iso_minimal_is_normative_binarize_and_thin_only(self) -> None:
        R = np.zeros((40, 40), dtype=np.uint8)
        R[5:35, 5:35] = 255
        vein = np.zeros((40, 40), dtype=np.uint8)
        vein[20, 10:30] = 255

        cfg = OpenVeinVesselCleanupConfig.iso_minimal()
        self.assertTrue(cfg.is_iso_minimal)
        self.assertEqual(cfg.preset_name, "iso_minimal")

        _, skel = prepare_vessel_skeleton(
            R,
            vein,
            vein_map_source="openvein_matlab",
            openvein_cleanup=cfg,
        )
        expected = zs_thinning(binarize_openvein_vein_map(vein, R == 255))
        np.testing.assert_array_equal(skel, expected)

    def test_heuristic_reduces_spur_noise_vs_iso_minimal(self) -> None:
        R = np.zeros((60, 60), dtype=np.uint8)
        R[:, :] = 255
        vein = np.zeros((60, 60), dtype=np.uint8)
        vein[30, 10:50] = 255
        # Short spurs on a thick segment (noise typical of PC maps).
        for x in range(12, 48, 4):
            vein[28:33, x] = 255

        _, skel_iso = prepare_vessel_skeleton(
            R,
            vein,
            vein_map_source="openvein_matlab",
            openvein_cleanup=OpenVeinVesselCleanupConfig.iso_minimal(),
        )
        _, skel_heur = prepare_vessel_skeleton(
            R,
            vein,
            vein_map_source="openvein_matlab",
            openvein_cleanup=OpenVeinVesselCleanupConfig.heuristic_default(),
        )
        self.assertGreater(int(skel_iso.sum()), int(skel_heur.sum()))

    def test_individual_heuristic_flags_toggle_steps(self) -> None:
        R = np.zeros((30, 30), dtype=np.uint8)
        R[:, :] = 255
        vein = np.zeros((30, 30), dtype=np.uint8)
        vein[14:17, 8:22] = 255

        base = OpenVeinVesselCleanupConfig.iso_minimal()
        with_open = OpenVeinVesselCleanupConfig(
            morphological_opening=True,
            open_ksize=5,
        )
        _, sk0 = prepare_vessel_skeleton(
            R, vein, vein_map_source="openvein_matlab", openvein_cleanup=base
        )
        _, sk1 = prepare_vessel_skeleton(
            R, vein, vein_map_source="openvein_matlab", openvein_cleanup=with_open
        )
        self.assertNotEqual(int(sk0.sum()), int(sk1.sum()))


if __name__ == "__main__":
    unittest.main()
