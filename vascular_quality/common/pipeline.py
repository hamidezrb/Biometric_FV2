"""ISO Q1–Q9 pipeline shared by anatomy-specific experiment runners."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from iso_constants import CaptureSite
from q1 import calculate_q1_detailed
from q2 import calculate_q2
from q3 import calculate_q3
from q4 import calculate_q4
from q5 import calculate_q5
from q6 import calculate_q6
from q7 import calculate_q7
from q8 import calculate_q8
from q9 import calculate_q9
from vessel_utils import OpenVeinVesselCleanupConfig, prepare_vessel_skeleton_with_stages
from unified_quality import (
    DEFAULT_POWER_COEFFICIENTS,
    evaluate_unified_quality,
)
from vascular_quality.common.openvein import vein_map_path
from vascular_quality.common.paths import ensure_dir
from vascular_quality.finger_vein.config import DEFAULT_OPENVEIN_VESSEL_CLEANUP
from vascular_quality.common.debug_outputs import (
    write_pc_vessel_debug_stages,
    write_q_debug_images,
)
from vascular_quality.common.visualization import (
    get_endpoints_intersections,
    visualize_feature_points_iso_style,
)


def run_q1_q9_on_image(
    image_path: Path | str,
    vein_root: Path | str,
    out_dir: Path | str | None = None,
    *,
    capture_site: CaptureSite = CaptureSite.FINGER_SECOND_PHALANX,
    save_debug_images: bool = False,
    openvein_cleanup: OpenVeinVesselCleanupConfig | None = None,
) -> dict[str, Any]:
    """
    Compute Q1–Q9 and unified quality for one vascular image.

    Args:
        image_path: Grayscale vascular capture.
        vein_root: Folder containing OpenVein vein maps (same filenames).
        out_dir: Where to write debug PNGs when ``save_debug_images=True``.
        capture_site: ISO Table 1–3 capture site for Sc, Lc, Fc.
        save_debug_images: Write intermediate visualization PNGs (off by default).
    """
    image_path = Path(image_path)
    vein_root = Path(vein_root)
    base = image_path.stem

    q1_result = calculate_q1_detailed(
        str(image_path),
        capture_site=capture_site,
    )
    Q1 = q1_result.Q1_score
    Sunocc = q1_result.S_unoccluded
    R_mask = q1_result.R_foreground
    gray = q1_result.grayscale
    R_unoccluded = q1_result.R_unoccluded

    Q2, cx, cy, S_H, S_V = calculate_q2(R_mask, gray)
    Q3, sigma, g_mean = calculate_q3(R_mask, gray)
    Q4, sigma, g_mean = calculate_q4(R_mask, gray)
    Q5, H_bits = calculate_q5(R_mask, gray, bit_depth=8, ep_c=0.75)
    Q6, N100 = calculate_q6(R_mask, gray, S_unoccluded=Sunocc, gc=0.006)
    Q7, block_var = calculate_q7(R_mask, gray, g_mean)

    vein_path = vein_map_path(vein_root, image_path)
    if not vein_path.is_file():
        raise FileNotFoundError(
            f"Vein map not found: {vein_path}\n"
            f"Expected one file per input image under {vein_root}."
        )

    vein_img = cv2.imread(str(vein_path), cv2.IMREAD_GRAYSCALE)
    if vein_img is None:
        raise ValueError(f"Could not read vein map: {vein_path}")

    if vein_img.shape != R_mask.shape:
        raise ValueError(
            f"Shape mismatch: vein_map={vein_img.shape} vs R_mask={R_mask.shape}. "
            f"Export vein maps at the same resolution as the original image."
        )

    cleanup = openvein_cleanup or DEFAULT_OPENVEIN_VESSEL_CLEANUP

    _vessel01, skel01, vessel_stages = prepare_vessel_skeleton_with_stages(
        R_mask,
        vein_img,
        vein_map_source="openvein_matlab",
        R_unoccluded=R_unoccluded,
        openvein_cleanup=cleanup,
    )

    Q8, N_vessel, skel_q8 = calculate_q8(
        R_mask,
        vein_img,
        capture_site=capture_site,
        vein_map_source="openvein_matlab",
        R_unoccluded=R_unoccluded,
        openvein_cleanup=cleanup,
        skel01=skel01,
    )
    Q9, N_fp, N_end, N_int, skel_q9 = calculate_q9(
        R_mask,
        vein_img,
        capture_site=capture_site,
        vein_map_source="openvein_matlab",
        R_unoccluded=R_unoccluded,
        openvein_cleanup=cleanup,
        skel01=skel01,
    )

    end_pts, int_pts = get_endpoints_intersections(skel_q9, R_mask == 255)
    q9_points_vis = visualize_feature_points_iso_style(skel_q9, end_pts, int_pts)

    debug_dir: str | None = None
    if save_debug_images:
        if out_dir is None:
            raise ValueError("out_dir is required when save_debug_images=True")
        debug_out = ensure_dir(out_dir)
        write_pc_vessel_debug_stages(
            debug_out,
            original=gray,
            stages=vessel_stages,
            skel_q8=skel_q8,
            q9_points_vis=q9_points_vis,
        )
        write_q_debug_images(
            debug_out,
            gray=gray,
            r_mask=R_mask,
            unoccluded_mask=R_unoccluded,
            vein_img=vein_img,
            skel_q8=skel_q8,
            skel_q9=skel_q9,
            q9_points_vis=q9_points_vis,
        )
        debug_dir = str(debug_out)

    qi = {
        "Q1": Q1, "Q2": Q2, "Q3": Q3, "Q4": Q4, "Q5": Q5,
        "Q6": Q6, "Q7": Q7, "Q8": Q8, "Q9": Q9,
    }
    unified = evaluate_unified_quality(qi, DEFAULT_POWER_COEFFICIENTS)

    return {
        "file": base,
        "image_path": str(image_path),
        "Q1": Q1, "Sunocc": Sunocc,
        "Q2": Q2,
        "Q3": Q3, "sigma": float(sigma), "g_mean": float(g_mean),
        "Q4": Q4,
        "Q5": Q5, "H_bits": float(H_bits),
        "Q6": Q6, "N100": int(N100),
        "Q7": Q7,
        "Q8": Q8, "N_vessel": int(N_vessel),
        "Q9": Q9, "N_fp": int(N_fp), "N_end": int(N_end), "N_int": int(N_int),
        "unified_quality_score": unified["unified_quality_score"],
        "power_coefficients": unified["power_coefficients"],
        "coefficients_are_placeholder": unified["coefficients_are_placeholder"],
        "vein_map": str(vein_path),
        "debug_dir": debug_dir,
        "capture_site": capture_site.value,
        "vessel_cleanup": cleanup.preset_name,
    }
