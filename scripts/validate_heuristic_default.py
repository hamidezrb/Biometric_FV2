"""Validate heuristic_default vessel cleanup on all existing finger-vein images."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from q1 import calculate_q1_detailed
from q8 import calculate_q8, count_vessel_pixels_in_foreground
from q9 import calculate_q9, count_feature_points_in_foreground
from vascular_quality.common.debug_outputs import (
    PC_VESSEL_DEBUG_FILENAMES,
    finger_vein_image_debug_dir,
)
from vascular_quality.common.openvein import vein_map_path
from vascular_quality.common.paths import (
    finger_vein_image_dir,
    iter_quality_classes,
    openvein_vein_map_dir,
)
from vascular_quality.common.images import list_images_in_dir
from vascular_quality.common.pipeline import run_q1_q9_on_image
from vascular_quality.finger_vein.config import (
    DEFAULT_CAPTURE_SITE,
    FINGER_VEIN_DATASETS,
)
from vessel_utils import (
    OpenVeinVesselCleanupConfig,
    prepare_vessel_skeleton,
)

REQUIRED_DEBUG = set(PC_VESSEL_DEBUG_FILENAMES)


def collect_jobs() -> list[tuple[str, str, Path]]:
    jobs: list[tuple[str, str, Path]] = []
    for dataset in FINGER_VEIN_DATASETS:
        for quality in iter_quality_classes("all"):
            image_dir = finger_vein_image_dir(dataset, quality)
            if not image_dir.is_dir():
                continue
            for image_path in list_images_in_dir(image_dir):
                vein_path = vein_map_path(
                    openvein_vein_map_dir(dataset, quality, "PC"), image_path
                )
                if vein_path.is_file():
                    jobs.append((dataset, quality, image_path))
    return jobs


def compare_skeletons(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.array_equal((a > 0).astype(np.uint8), (b > 0).astype(np.uint8)))


def main() -> int:
    heuristic = OpenVeinVesselCleanupConfig.heuristic_default()
    iso = OpenVeinVesselCleanupConfig.iso_minimal()
    jobs = collect_jobs()
    print(f"Found {len(jobs)} image(s) with PC vein maps.\n")

    rows_heur: list[dict] = []
    rows_compare: list[dict] = []

    for dataset, quality, image_path in jobs:
        vein_root = openvein_vein_map_dir(dataset, quality, "PC")
        debug_dir = finger_vein_image_debug_dir(dataset, quality, image_path.stem)

        result = run_q1_q9_on_image(
            image_path,
            vein_root,
            debug_dir,
            save_debug_images=True,
            capture_site=DEFAULT_CAPTURE_SITE,
            openvein_cleanup=heuristic,
        )

        # ISO-minimal baseline for comparison
        q1 = calculate_q1_detailed(str(image_path), capture_site=DEFAULT_CAPTURE_SITE)
        vein = cv2.imread(str(vein_map_path(vein_root, image_path)), cv2.IMREAD_GRAYSCALE)
        R = q1.R_foreground
        fg = R == 255
        _, sk_iso = prepare_vessel_skeleton(
            R, vein, vein_map_source="openvein_matlab",
            R_unoccluded=q1.R_unoccluded, openvein_cleanup=iso,
        )
        _, sk_heur = prepare_vessel_skeleton(
            R, vein, vein_map_source="openvein_matlab",
            R_unoccluded=q1.R_unoccluded, openvein_cleanup=heuristic,
        )
        Q8_iso, Nv_iso, _ = calculate_q8(
            R, vein, vein_map_source="openvein_matlab",
            R_unoccluded=q1.R_unoccluded, openvein_cleanup=iso,
        )
        Q9_iso, fp_iso, ne_iso, ni_iso, _ = calculate_q9(
            R, vein, vein_map_source="openvein_matlab",
            R_unoccluded=q1.R_unoccluded, openvein_cleanup=iso,
        )

        # Debug file checks
        missing = REQUIRED_DEBUG - {p.name for p in debug_dir.glob("*.png")}
        q8_png = cv2.imread(str(debug_dir / "q8_skeleton_used.png"), cv2.IMREAD_GRAYSCALE)
        pc_skel_png = cv2.imread(str(debug_dir / "pc_skeleton.png"), cv2.IMREAD_GRAYSCALE)
        q8_matches = compare_skeletons(
            (result["N_vessel"] and sk_heur) or sk_heur,
            q8_png if q8_png is not None else np.zeros(1),
        )
        skel_match = compare_skeletons(sk_heur, pc_skel_png) if pc_skel_png is not None else False
        q8_skel_match = compare_skeletons(sk_heur, q8_png) if q8_png is not None else False

        rows_heur.append({
            "dataset": dataset,
            "quality": quality,
            "image": image_path.name,
            "vessel_cleanup": result.get("vessel_cleanup"),
            "Q8": result["Q8"],
            "Q9": result["Q9"],
            "N_vessel": result["N_vessel"],
            "N_end": result["N_end"],
            "N_int": result["N_int"],
            "N_fp": result["N_fp"],
            "unified": result["unified_quality_score"],
            "debug_ok": len(missing) == 0,
            "skel_pc_q8_match": skel_match and q8_skel_match,
        })

        rows_compare.append({
            "dataset": dataset,
            "quality": quality,
            "image": image_path.name,
            "skel_iso": int(sk_iso.sum()),
            "skel_heur": int(sk_heur.sum()),
            "end_iso": ne_iso,
            "end_heur": result["N_end"],
            "int_iso": ni_iso,
            "int_heur": result["N_int"],
            "Q8_iso": Q8_iso,
            "Q8_heur": result["Q8"],
            "Q9_iso": Q9_iso,
            "Q9_heur": result["Q9"],
            "Nv_iso": Nv_iso,
            "Nv_heur": result["N_vessel"],
            "fp_iso": fp_iso,
            "fp_heur": result["N_fp"],
        })

        status = "OK" if not missing and q8_skel_match else "ISSUE"
        print(f"[{status}] {dataset}/{quality}/{image_path.name}")
        if missing:
            print(f"  missing debug: {sorted(missing)}")
        if not q8_skel_match:
            print("  WARNING: q8_skeleton_used.png != computed skeleton")

    out_dir = _PROJECT_ROOT / "results" / "finger_vein" / "PC_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    df_heur = pd.DataFrame(rows_heur)
    df_cmp = pd.DataFrame(rows_compare)
    df_heur.to_csv(out_dir / "validation_heuristic.csv", index=False)
    df_cmp.to_csv(out_dir / "validation_compare_iso_vs_heuristic.csv", index=False)

    print("\n--- Heuristic results ---")
    print(df_heur.to_string(index=False))
    print("\n--- ISO vs heuristic comparison ---")
    print(df_cmp.to_string(index=False))

    dupes = df_heur.duplicated(subset=["dataset", "quality", "image"]).sum()
    print(f"\nDuplicate rows: {dupes}")
    print(f"vessel_cleanup values: {df_heur['vessel_cleanup'].unique().tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
