"""
ISO/IEC 29794-9 Clause 5.2.1 — Q1 re-audit with debug mask outputs.

Generates per-image:
  - foreground mask
  - occlusion mask
  - unoccluded foreground mask
  - pixel counts and Formula (1) breakdown
  - old-vs-new foreground comparison (Q8/Q9 impact)
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
from typing import Dict, List, Tuple

import cv2
import numpy as np

from iso_constants import CaptureSite, get_capture_site_coefficients
from iso_foreground import extract_foreground_region
from q1 import calculate_q1_detailed, compute_q1_score
from q8 import calculate_q8
from q9 import calculate_q9


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def overlay_mask(gray: np.ndarray, mask255: np.ndarray, color=(0, 255, 0)) -> np.ndarray:
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    fg = mask255 == 255
    if np.any(fg):
        vis[fg] = (0.6 * vis[fg] + 0.4 * np.array(color)).astype(np.uint8)
    return vis


def extract_foreground_region_legacy_contour(grayscale: np.ndarray) -> Tuple[np.ndarray, int]:
    """Pre-refactor foreground: Otsu + largest external contour fill (no polarity fix)."""
    _, binary_mask = cv2.threshold(
        grayscale, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return np.zeros_like(grayscale, dtype=np.uint8), 0

    largest_contour = max(contours, key=cv2.contourArea)
    R_mask = np.zeros_like(binary_mask, dtype=np.uint8)
    cv2.fillPoly(R_mask, [largest_contour], 255)
    S_foreground = int(np.count_nonzero(R_mask == 255))
    return R_mask, S_foreground


def extract_foreground_region_broken_polarity(grayscale: np.ndarray) -> Tuple[np.ndarray, int]:
    """Broken refactor: Otsu + invert-if-larger + largest CC (background-as-fg bug)."""
    gray = grayscale if grayscale.dtype == np.uint8 else grayscale.astype(np.uint8)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask_inv = cv2.bitwise_not(mask)
    if int(np.sum(mask)) < int(np.sum(mask_inv)):
        mask = mask_inv

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return np.zeros_like(gray, dtype=np.uint8), 0

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + int(np.argmax(areas))
    R_mask = np.where(labels == largest_label, 255, 0).astype(np.uint8)
    return R_mask, int(np.count_nonzero(R_mask == 255))


def mask_symmetric_difference(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero((a == 255) ^ (b == 255)))


def audit_image(
    image_path: str,
    out_dir: str,
    vein_root: str,
    capture_site: CaptureSite,
) -> Dict[str, object]:
    ensure_dir(out_dir)
    base = os.path.splitext(os.path.basename(image_path))[0]

    result = calculate_q1_detailed(image_path, capture_site=capture_site)
    gray = result.grayscale

    R_old, S_old = extract_foreground_region_legacy_contour(gray)
    R_broken, S_broken = extract_foreground_region_broken_polarity(gray)
    R_new = result.R_foreground
    S_new = result.S_foreground

    compliant = result.S_unoccluded == (result.S_foreground - result.S_occluded)
    used_fg_as_effective = result.S_unoccluded == result.S_foreground and result.S_occluded > 0

    # Debug masks
    cv2.imwrite(os.path.join(out_dir, f"{base}_foreground_mask.png"), R_new)
    cv2.imwrite(os.path.join(out_dir, f"{base}_occlusion_mask.png"), result.occlusion_mask)
    cv2.imwrite(os.path.join(out_dir, f"{base}_unoccluded_mask.png"), result.R_unoccluded)
    cv2.imwrite(os.path.join(out_dir, f"{base}_overlay_foreground.png"), overlay_mask(gray, R_new))
    cv2.imwrite(
        os.path.join(out_dir, f"{base}_overlay_occlusion.png"),
        overlay_mask(gray, result.occlusion_mask, color=(0, 0, 255)),
    )
    cv2.imwrite(
        os.path.join(out_dir, f"{base}_overlay_unoccluded.png"),
        overlay_mask(gray, result.R_unoccluded, color=(0, 255, 255)),
    )

    vein_path = os.path.join(vein_root, os.path.basename(image_path))
    vein_img = cv2.imread(vein_path, cv2.IMREAD_GRAYSCALE)

    q8_q9: Dict[str, object] = {}
    if vein_img is not None and vein_img.shape == R_new.shape:
        Q8_old, Nv_old, _ = calculate_q8(R_old, vein_img, capture_site=capture_site)
        Q9_old, Nfp_old, Nend_old, Nint_old, _ = calculate_q9(R_old, vein_img, capture_site=capture_site)
        Q8_new, Nv_new, _ = calculate_q8(R_new, vein_img, capture_site=capture_site)
        Q9_new, Nfp_new, Nend_new, Nint_new, _ = calculate_q9(R_new, vein_img, capture_site=capture_site)

        q8_q9 = {
            "Q8_old": Q8_old,
            "Q8_new": Q8_new,
            "N_vessel_old": Nv_old,
            "N_vessel_new": Nv_new,
            "Q9_old": Q9_old,
            "Q9_new": Q9_new,
            "N_fp_old": Nfp_old,
            "N_fp_new": Nfp_new,
            "N_end_old": Nend_old,
            "N_end_new": Nend_new,
            "N_int_old": Nint_old,
            "N_int_new": Nint_new,
        }

    row = {
        "file": base,
        "compliance": "NOT COMPLIANT" if used_fg_as_effective else "COMPLIANT",
        "S_foreground": result.S_foreground,
        "S_occluded": result.S_occluded,
        "S_unoccluded": result.S_unoccluded,
        "Sc": result.Sc,
        "q1_raw": round(result.q1_raw, 4),
        "Q1": result.Q1_score,
        "Q1_if_fg_only": compute_q1_score(result.S_foreground, result.Sc),
        "Q1_broken_polarity": compute_q1_score(
            S_broken - int(np.count_nonzero((result.occlusion_mask == 255) & (R_broken == 255))),
            result.Sc,
        ),
        "Q1_legacy_contour": compute_q1_score(
            S_old - int(np.count_nonzero((result.occlusion_mask == 255) & (R_old == 255))),
            result.Sc,
        ),
        "S_foreground_legacy": S_old,
        "S_foreground_broken": S_broken,
        "S_foreground_new": S_new,
        "mask_symdiff_legacy_vs_new": mask_symmetric_difference(R_old, R_new),
        "mask_only_legacy": int(np.count_nonzero((R_old == 255) & (R_new == 0))),
        "mask_only_new": int(np.count_nonzero((R_old == 0) & (R_new == 255))),
        **q8_q9,
    }
    return row


def collect_image_paths(input_path: str | None) -> List[str]:
    if input_path is None:
        paths: List[str] = []
        for folder in ("test_images/high_quality", "test_images/low_quality"):
            if os.path.isdir(folder):
                paths.extend(sorted(glob.glob(os.path.join(folder, "*.*"))))
        return paths
    if os.path.isdir(input_path):
        return sorted(glob.glob(os.path.join(input_path, "*.*")))
    return [input_path]


def main() -> None:
    parser = argparse.ArgumentParser(description="Q1 Clause 5.2.1 compliance audit")
    parser.add_argument("--input", default=None, help="Image file or folder")
    parser.add_argument("--out", default="debug_outputs_q1_audit", help="Debug output directory")
    parser.add_argument(
        "--vein_root",
        default="debug_openvein_features/RLT",
        help="Folder with vein maps for Q8/Q9 comparison",
    )
    parser.add_argument(
        "--capture-site",
        default="finger_second_phalanx",
        choices=[site.value for site in CaptureSite],
        help="Table 1 capture site for Sc",
    )
    args = parser.parse_args()

    capture_site = CaptureSite(args.capture_site)
    paths = collect_image_paths(args.input)
    if not paths:
        parser.error("No images found")

    ensure_dir(args.out)
    rows: List[Dict[str, object]] = []

    print("\nISO/IEC 29794-9 Clause 5.2.1 — Q1 Re-Audit\n")
    print(f"Capture site: {capture_site.value}")
    print(f"Sc = {get_capture_site_coefficients(capture_site)['SC']}\n")

    for path in paths:
        try:
            row = audit_image(path, args.out, args.vein_root, capture_site)
            rows.append(row)
            print(f"=== {row['file']} ===")
            print(f"  Compliance (effective = fg - occ): {row['compliance']}")
            print(f"  S_foreground:  {row['S_foreground']}")
            print(f"  S_occluded:    {row['S_occluded']}")
            print(f"  S_unoccluded:  {row['S_unoccluded']}  (Formula 1 numerator)")
            print(f"  Sc:            {row['Sc']}")
            print(f"  q1_raw:        {row['q1_raw']}")
            print(f"  Q1:            {row['Q1']}")
            print(f"  Q1 if fg only: {row['Q1_if_fg_only']}  (non-compliant shortcut)")
            print(f"  Legacy contour S_fg={row['S_foreground_legacy']} Q1={row['Q1_legacy_contour']}")
            print(f"  Broken polarity S_fg={row['S_foreground_broken']} Q1={row['Q1_broken_polarity']}")
            print(
                f"  Mask symdiff legacy vs new: {row['mask_symdiff_legacy_vs_new']} "
                f"(only_legacy={row['mask_only_legacy']}, only_new={row['mask_only_new']})"
            )
            if "Q8_old" in row:
                print(
                    f"  Q8: {row['Q8_old']} -> {row['Q8_new']}  "
                    f"(N_vessel {row['N_vessel_old']} -> {row['N_vessel_new']})"
                )
                print(
                    f"  Q9: {row['Q9_old']} -> {row['Q9_new']}  "
                    f"(N_fp {row['N_fp_old']} -> {row['N_fp_new']})"
                )
            print()
        except Exception as exc:
            print(f"ERROR on {path}: {exc}\n")

    if rows:
        csv_path = os.path.join(args.out, "q1_audit_summary.csv")
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Summary CSV: {csv_path}")
        print(f"Debug masks: {args.out}/")


if __name__ == "__main__":
    main()
