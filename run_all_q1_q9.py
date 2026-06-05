import os
import glob
import cv2
import numpy as np

from tabulate import tabulate

from q1 import calculate_q1
from q2 import calculate_q2
from q3 import calculate_q3
from q4 import calculate_q4
from q5 import calculate_q5
from q6 import calculate_q6
from q7 import calculate_q7

# ISO-aligned versions:
# Q8: calculate_q8(R_mask, vein_img_or_path, Lc=600) -> (Q8, N_vessel, skeleton01)
# Q9: calculate_q9(R_mask, vein_img_or_path, Fc=15) -> (Q9, N_fp, N_end, N_int, skeleton01)
from q8 import calculate_q8
from q9 import calculate_q9
import vessel_utils
from unified_quality import (
    DEFAULT_POWER_COEFFICIENTS,
    evaluate_unified_quality,
    format_power_coefficients,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TEST_IMAGE_DIRS = (
    os.path.join(SCRIPT_DIR, "test_images", "high_quality"),
    os.path.join(SCRIPT_DIR, "test_images", "low_quality"),
)
DEFAULT_VEIN_ROOT = os.path.join(SCRIPT_DIR, "debug_openvein_features", "RLT")


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def overlay_mask(gray: np.ndarray, mask255: np.ndarray) -> np.ndarray:
    """Return a 3-channel visualization: gray + green foreground overlay."""
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    fg = (mask255 == 255)
    vis[fg] = (0.6 * vis[fg] + 0.4 * np.array([0, 255, 0])).astype(np.uint8)
    return vis

def get_endpoints_intersections(skel01: np.ndarray, foreground_mask=None):
    """Return lists of (x,y) endpoints and intersections from a 0/1 skeleton."""
    from q9 import transition_count_8nbr

    H, W = skel01.shape
    if foreground_mask is None:
        foreground_mask = np.ones((H, W), dtype=bool)
    endpoints = []
    intersections = []
    for y in range(H):
        for x in range(W):
            if not foreground_mask[y, x] or skel01[y, x] != 1:
                continue
            T = transition_count_8nbr(skel01, y, x)
            if T == 2:
                endpoints.append((x, y))
            elif T in (6, 8):
                intersections.append((x, y))
    return endpoints, intersections

def visualize_feature_points_iso_style(skel01: np.ndarray, endpoints, intersections) -> np.ndarray:
    """
    ISO-style overlay:
    - endpoints: red circle outline
    - intersections: red square outline
    Drawn on top of the skeleton.
    """
    vis = cv2.cvtColor((skel01.astype(np.uint8) * 255), cv2.COLOR_GRAY2BGR)

    red = (0, 0, 255)  # BGR

    # endpoints: red circles (outline)
    for (x, y) in endpoints:
        cv2.circle(vis, (x, y), 3, red, 1)  # radius=3, thickness=1

    # intersections: red squares (outline)
    for (x, y) in intersections:
        cv2.rectangle(vis, (x - 3, y - 3), (x + 3, y + 3), red, 1)  # thickness=1

    return vis



def run_on_image(image_path: str, vein_root: str, out_dir: str):
    ensure_dir(out_dir)
    base = os.path.splitext(os.path.basename(image_path))[0]

    # --- Q1
    Q1, Sunocc, R_mask, gray = calculate_q1(image_path)

    # --- Q2
    Q2, cx, cy, S_H, S_V = calculate_q2(R_mask, gray)

    # --- Q3 + Q4 + Q6  (disable q8 inside q3 if your q3 bundles it)
    # Q3, Q4, Q6, _dummy_q8, sigma, g_mean, N100, _dummy_nv = calculate_q3(
    #     R_mask, gray, Sunocc, include_q8=False
    # )
    
    
    Q3, sigma, g_mean = calculate_q3(R_mask, gray)
    Q4, sigma, g_mean = calculate_q4(R_mask, gray)
    Q5, H_bits = calculate_q5(R_mask, gray, bit_depth=8, ep_c=0.75)
    Q6, N100 = calculate_q6(R_mask, gray, S_unoccluded=Sunocc, gc=0.006)

    # --- Q5
    # Q5, H_bits = calculate_q5(R_mask, gray, bit_depth=8, ep_c=0.75)

    # --- Q7
    Q7, block_var = calculate_q7(R_mask, gray, g_mean)

    # --- Find MATLAB vein extraction map (binary vessel image)
    vein_path = os.path.join(vein_root, os.path.basename(image_path))
    vein_img = cv2.imread(vein_path, cv2.IMREAD_GRAYSCALE)
    if vein_img is None:
        raise ValueError(f"Could not read vein map: {vein_path}")

    # Ensure same size as original/R_mask (ISO expects alignment)
    if vein_img.shape != R_mask.shape:
        raise ValueError(
            f"Shape mismatch: vein_map={vein_img.shape} vs R_mask={R_mask.shape}. "
            f"Fix by exporting MATLAB vein map in the same resolution as the original image."
        )
        
        
    # --- RAW skeleton (before filtering) for visualization ---
    raw_binary = ((vein_img > 0) & (R_mask == 255)).astype(np.uint8)

    # Use the same thinning algorithm used in Q8/Q9
    skel_raw = vessel_utils.zs_thinning(raw_binary)

    # --- Q8 / Q9 (ISO-minimal skeleton path; no per-dataset tuning)
    Q8, N_vessel, skel_q8 = calculate_q8(R_mask, vein_img)
    Q9, N_fp, N_end, N_int, skel_q9 = calculate_q9(R_mask, vein_img)
    
    # For visualization/debugging: extract endpoint and intersection coordinates from the skeleton
    end_pts, int_pts = get_endpoints_intersections(skel_q9, R_mask == 255)
    q9_points_vis = visualize_feature_points_iso_style(skel_q9, end_pts, int_pts)

    # --- Save debug images
    cv2.imwrite(os.path.join(out_dir, f"{base}_gray.png"), gray)
    
    # cv2.imwrite(os.path.join(out_dir, f"{base}_Rmask.png"), R_mask)
    
    cv2.imwrite(os.path.join(out_dir, f"{base}_overlay_R.png"), overlay_mask(gray, R_mask))

    cv2.imwrite(os.path.join(out_dir, f"{base}_vein_map_matlab.png"), vein_img)
    
    # cv2.imwrite(os.path.join(out_dir, f"{base}_q8_skeleton.png"), (skel_q8.astype(np.uint8) * 255))
    
    # skeleton before removing disconnected parts
    cv2.imwrite(os.path.join(out_dir, f"{base}_q9_skeleton_raw.png"),(skel_raw.astype(np.uint8) * 255))
    # skeleton after cleaning/pruning
    cv2.imwrite(os.path.join(out_dir, f"{base}_q9_skeleton_filtered.png"), (skel_q9.astype(np.uint8) * 255))
    # endpoints (red circles) + intersections (red squares)
    cv2.imwrite(os.path.join(out_dir, f"{base}_q9_points_vis.png"), q9_points_vis)

    qi = {
        "Q1": Q1, "Q2": Q2, "Q3": Q3, "Q4": Q4, "Q5": Q5,
        "Q6": Q6, "Q7": Q7, "Q8": Q8, "Q9": Q9,
    }
    unified = evaluate_unified_quality(qi, DEFAULT_POWER_COEFFICIENTS)

    result = {
        "file": base,
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
        "vein_map": vein_path,
        "debug_dir": out_dir,
    }
    return result


def collect_default_image_paths() -> list[str]:
    paths: list[str] = []
    for folder in DEFAULT_TEST_IMAGE_DIRS:
        if os.path.isdir(folder):
            paths.extend(sorted(glob.glob(os.path.join(folder, "*.*"))))
    return paths


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        default=None,
        help="Path to an image OR a folder of images (default: all test_images/*)",
    )
    ap.add_argument(
        "--vein_root",
        default=DEFAULT_VEIN_ROOT,
        help="Folder with MATLAB vein maps for one extractor (default: debug_openvein_features/RLT)",
    )
    ap.add_argument("--out", default="debug_outputs", help="Where to save debug images")
    args = ap.parse_args()

    if args.input is None:
        paths = collect_default_image_paths()
        if not paths:
            ap.error("No default test images found under test_images/")
    elif os.path.isdir(args.input):
        paths = sorted(glob.glob(os.path.join(args.input, "*.*")))
    else:
        paths = [args.input]

    ensure_dir(args.out)

    all_results = []

    headers = [
        "image",
        "Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8", "Q9",
        "unified_quality_score",
        "N_vessel", "N_end", "N_int", "N_fp",
    ]
    for p in paths:
        try:
            r = run_on_image(p, args.vein_root, args.out)
            all_results.append(r)
        except Exception as e:
            print(f"ERROR on {p}: {e}")

    table = []
    for r in all_results:
        table.append([
            r["file"],
            r["Q1"], r["Q2"], r["Q3"], r["Q4"], r["Q5"],
            r["Q6"], r["Q7"], r["Q8"], r["Q9"],
            r["unified_quality_score"],
            r["N_vessel"], r["N_end"], r["N_int"], r["N_fp"],
        ])

    coeffs = (
        all_results[0]["power_coefficients"]
        if all_results
        else DEFAULT_POWER_COEFFICIENTS
    )

    print("\nQUALITY SCORES (ISO/IEC 29794-9)\n")
    print("Power coefficients w_i (ISO 5.3):")
    print(f"  {format_power_coefficients(coeffs)}")
    if coeffs == DEFAULT_POWER_COEFFICIENTS:
        print("  (placeholder 1/9 each — TODO: import calibrated w_i from ISO reference implementation)\n")
    else:
        print()
    if table:
        print(tabulate(table, headers=headers, tablefmt="github"))
    else:
        print("No images processed successfully.")

if __name__ == "__main__":
    main()
