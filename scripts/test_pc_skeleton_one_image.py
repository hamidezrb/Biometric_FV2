"""Compare PC skeleton preprocessing before/after on one image."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from q1 import calculate_q1_detailed
from q8 import calculate_q8, count_vessel_pixels_in_foreground
from q9 import calculate_q9, count_feature_points_in_foreground
from vascular_quality.common.debug_outputs import finger_vein_image_debug_dir
from vascular_quality.common.openvein import vein_map_path
from vascular_quality.common.paths import finger_vein_image_dir, openvein_vein_map_dir
from vascular_quality.common.pipeline import run_q1_q9_on_image
from vascular_quality.finger_vein.config import DEFAULT_CAPTURE_SITE
from vessel_utils import prepare_vessel_skeleton


def main() -> int:
    dataset, quality = "SCUT", "low_quality"
    image_name = "38_5_1_5_3.bmp"
    image_path = finger_vein_image_dir(dataset, quality) / image_name
    vein_root = openvein_vein_map_dir(dataset, quality, "PC")
    vein_path = vein_map_path(vein_root, image_path)
    vein = cv2.imread(str(vein_path), cv2.IMREAD_GRAYSCALE)
    q1 = calculate_q1_detailed(str(image_path))
    R = q1.R_foreground
    fg = R == 255

    _, sk_old = prepare_vessel_skeleton(R, vein, vein_map_source="iso")
    Q8_old, Nv_old, _ = calculate_q8(R, vein, vein_map_source="iso")
    Q9_old, Nfp_old, Ne_old, Ni_old, _ = calculate_q9(R, vein, vein_map_source="iso")

    debug_dir = finger_vein_image_debug_dir(dataset, quality, image_path.stem)
    result = run_q1_q9_on_image(
        image_path,
        vein_root,
        debug_dir,
        save_debug_images=True,
        capture_site=DEFAULT_CAPTURE_SITE,
    )

    _, sk_new = prepare_vessel_skeleton(
        R, vein, vein_map_source="openvein_matlab", R_unoccluded=q1.R_unoccluded
    )
    Nv_new = count_vessel_pixels_in_foreground(sk_new, fg)
    Ne_new, Ni_new, Nfp_new = count_feature_points_in_foreground(sk_new, fg)

    print("Image:", image_path)
    print("Debug dir:", debug_dir)
    print()
    print("OLD (iso binarize + thin only)")
    print("  skeleton pixels:", int(sk_old.sum()))
    print("  Q8:", Q8_old, "N_vessel:", Nv_old)
    print("  Q9:", Q9_old, "N_fp:", Nfp_old, "end:", Ne_old, "int:", Ni_old)
    print()
    print("NEW (openvein_matlab clean + thin)")
    print("  skeleton pixels:", int(sk_new.sum()))
    print("  Q8:", result["Q8"], "N_vessel:", result["N_vessel"])
    print("  Q9:", result["Q9"], "N_fp:", result["N_fp"], "end:", result["N_end"], "int:", result["N_int"])
    print()
    print("Debug stages written:", [p.name for p in sorted(debug_dir.glob("*.png"))])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
