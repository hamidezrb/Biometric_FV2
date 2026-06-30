"""
Run ISO Q1–Q7 on vascular images (no Q8/Q9 / unified score).

For full Q1–Q9 use run_finger_vein_experiment.py or run_all_q1_q9.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from tabulate import tabulate

from q1 import calculate_q1
from q2 import calculate_q2
from q3 import calculate_q3
from q4 import calculate_q4
from q5 import calculate_q5
from q6 import calculate_q6
from q7 import calculate_q7
from vascular_quality.common.images import list_images_in_dir
from vascular_quality.common.paths import ensure_dir, finger_vein_image_dir, iter_quality_classes
from vascular_quality.common.visualization import overlay_mask
from vascular_quality.finger_vein.config import DEFAULT_CAPTURE_SITE, FINGER_VEIN_DATASETS


def run_on_image(image_path: Path, out_dir: Path) -> dict:
    ensure_dir(out_dir)
    base = image_path.stem

    Q1, Sunocc, R_mask, gray = calculate_q1(
        str(image_path),
        capture_site=DEFAULT_CAPTURE_SITE,
    )
    Q2, _, _, _, _ = calculate_q2(R_mask, gray)
    Q3, sigma, g_mean = calculate_q3(R_mask, gray)
    Q4, _, _ = calculate_q4(R_mask, gray)
    Q5, _ = calculate_q5(R_mask, gray, bit_depth=8, ep_c=0.75)
    Q6, _ = calculate_q6(R_mask, gray, S_unoccluded=Sunocc, gc=0.006)
    Q7, block_var = calculate_q7(R_mask, gray, g_mean)

    cv2.imwrite(str(out_dir / f"{base}_gray.png"), gray)
    cv2.imwrite(str(out_dir / f"{base}_Rmask.png"), R_mask)
    cv2.imwrite(str(out_dir / f"{base}_overlay_R.png"), overlay_mask(gray, R_mask))

    return {
        "file": base,
        "Q1": int(Q1),
        "Q2": int(Q2),
        "Q3": int(Q3),
        "Q4": int(Q4),
        "Q5": int(Q5),
        "Q6": int(Q6),
        "Q7": int(Q7),
        "block_var": float(block_var),
        "debug_dir": str(out_dir),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="ISO Q1–Q7 pipeline")
    parser.add_argument("--input", default=None, help="Image file or folder")
    parser.add_argument(
        "--dataset",
        default="PLUS",
        choices=list(FINGER_VEIN_DATASETS),
        help="Dataset when --input omitted (default: PLUS).",
    )
    parser.add_argument(
        "--quality",
        default="all",
        choices=["high_quality", "low_quality", "all"],
    )
    parser.add_argument(
        "--out",
        default="debug_outputs_q1_q7",
        help="Debug output directory",
    )
    args = parser.parse_args()

    if args.input:
        if Path(args.input).is_dir():
            paths = list_images_in_dir(args.input)
        else:
            paths = [Path(args.input)]
    else:
        paths = []
        for q in iter_quality_classes(args.quality):
            paths.extend(list_images_in_dir(finger_vein_image_dir(args.dataset, q)))

    if not paths:
        parser.error("No images found. Pass --input or populate data/finger_vein/.")

    out_dir = ensure_dir(args.out)
    all_results = []
    for p in paths:
        try:
            all_results.append(run_on_image(p, out_dir))
        except Exception as exc:
            print(f"ERROR on {p}: {exc}")

    headers = ["image", "Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7"]
    table = [[r["file"], r["Q1"], r["Q2"], r["Q3"], r["Q4"], r["Q5"], r["Q6"], r["Q7"]]
             for r in all_results]

    print("\nQUALITY SCORES (ISO/IEC 29794-9) — Q1 to Q7\n")
    if table:
        print(tabulate(table, headers=headers, tablefmt="github"))
    else:
        print("No images processed successfully.")
    return 0 if all_results else 1


if __name__ == "__main__":
    raise SystemExit(main())
