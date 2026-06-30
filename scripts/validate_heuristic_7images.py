"""Re-validate heuristic_default on all existing finger-vein + PC images."""

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
from q8 import calculate_q8
from q9 import calculate_q9
from vascular_quality.common.debug_outputs import finger_vein_image_debug_dir
from vascular_quality.common.images import list_images_in_dir
from vascular_quality.common.openvein import vein_map_path
from vascular_quality.common.paths import (
    finger_vein_image_dir,
    iter_quality_classes,
    openvein_vein_map_dir,
)
from vascular_quality.finger_vein.config import (
    DEFAULT_CAPTURE_SITE,
    FINGER_VEIN_DATASETS,
)
from vessel_utils import (
    OpenVeinVesselCleanupConfig,
    binarize_openvein_vein_map,
    prepare_vessel_skeleton_with_stages,
)

# Import pipeline without triggering finger_vein package __init__ circular import.
import importlib.util

_pipeline_path = _PROJECT_ROOT / "vascular_quality" / "common" / "pipeline.py"
_spec = importlib.util.spec_from_file_location("vq_pipeline", _pipeline_path)
assert _spec and _spec.loader
_pipeline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pipeline)
run_q1_q9_on_image = _pipeline.run_q1_q9_on_image


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


def main() -> int:
    cleanup = OpenVeinVesselCleanupConfig.heuristic_default()
    jobs = collect_jobs()
    rows: list[dict] = []

    print(f"Preset: {cleanup.preset_name}")
    print(f"remove_border_touching={cleanup.remove_border_touching}, roi_margin={cleanup.roi_margin}\n")

    for dataset, quality, image_path in jobs:
        vein_root = openvein_vein_map_dir(dataset, quality, "PC")
        vein_path = vein_map_path(vein_root, image_path)
        vein = cv2.imread(str(vein_path), cv2.IMREAD_GRAYSCALE)
        q1 = calculate_q1_detailed(str(image_path), capture_site=DEFAULT_CAPTURE_SITE)
        Ru = q1.R_unoccluded == 255
        raw_pc = int(np.count_nonzero(binarize_openvein_vein_map(vein, Ru)))

        debug_dir = finger_vein_image_debug_dir(dataset, quality, image_path.stem)
        result = run_q1_q9_on_image(
            image_path,
            vein_root,
            debug_dir,
            save_debug_images=True,
            capture_site=DEFAULT_CAPTURE_SITE,
            openvein_cleanup=cleanup,
        )

        vessel01, skel01, stages = prepare_vessel_skeleton_with_stages(
            q1.R_foreground,
            vein,
            vein_map_source="openvein_matlab",
            R_unoccluded=q1.R_unoccluded,
            openvein_cleanup=cleanup,
        )
        after_clean = int(np.count_nonzero(vessel01))
        skel_px = int(np.count_nonzero(skel01))

        rows.append({
            "dataset": dataset,
            "quality": quality,
            "image": image_path.name,
            "raw_pc_pixels": raw_pc,
            "after_cleaning_pixels": after_clean,
            "skeleton_pixels": skel_px,
            "endpoints": result["N_end"],
            "intersections": result["N_int"],
            "Q8": result["Q8"],
            "Q9": result["Q9"],
            "N_vessel": result["N_vessel"],
            "N_fp": result["N_fp"],
            "unified_score": result["unified_quality_score"],
            "vessel_cleanup": result.get("vessel_cleanup"),
        })

        ok = skel_px > 0 and result["Q8"] > 0 and result["Q9"] > 0
        print(f"[{'OK' if ok else 'FAIL'}] {dataset}/{quality}/{image_path.name}  "
              f"raw={raw_pc} clean={after_clean} skel={skel_px} Q8={result['Q8']} Q9={result['Q9']}")

    df = pd.DataFrame(rows)
    out = _PROJECT_ROOT / "results" / "finger_vein" / "PC_validation"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "heuristic_fix_validation.csv", index=False)
    print("\n" + df.to_string(index=False))

    zero = df[df["skeleton_pixels"] == 0]
    print(f"\nImages with skeleton_pixels=0: {len(zero)}")
    if len(zero):
        print(zero[["dataset", "quality", "image", "raw_pc_pixels"]].to_string(index=False))

    return 0 if len(zero) == 0 and (df[["Q8", "Q9"]] > 0).all().all() else 1


if __name__ == "__main__":
    raise SystemExit(main())
