import os
import glob
import cv2
import numpy as np

from tabulate import tabulate

from q1 import calculate_q1
from q2 import calculate_q2
from q3 import calculate_q3
from q5 import calculate_q5
from q7 import calculate_q7

# ISO-aligned versions:
# Q8: calculate_q8(R_mask, vein_img_or_path, Lc=600) -> (Q8, N_vessel, skeleton01)
# Q9: calculate_q9(R_mask, vein_img_or_path, Fc=15) -> (Q9, N_fp, N_end, N_int, skeleton01)
from q8 import calculate_q8
from q9 import calculate_q9
import vessel_utils


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def overlay_mask(gray: np.ndarray, mask255: np.ndarray) -> np.ndarray:
    """Return a 3-channel visualization: gray + green foreground overlay."""
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    fg = (mask255 == 255)
    vis[fg] = (0.6 * vis[fg] + 0.4 * np.array([0, 255, 0])).astype(np.uint8)
    return vis

def transition_count_8nbr(skel01: np.ndarray, y: int, x: int) -> int:
    """Count 0<->1 transitions around the 8-neighborhood (ISO rule)."""
    p2 = skel01[y - 1, x]
    p3 = skel01[y - 1, x + 1]
    p4 = skel01[y, x + 1]
    p5 = skel01[y + 1, x + 1]
    p6 = skel01[y + 1, x]
    p7 = skel01[y + 1, x - 1]
    p8 = skel01[y, x - 1]
    p9 = skel01[y - 1, x - 1]
    seq = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
    return int(sum(seq[i] != seq[i + 1] for i in range(8)))

def get_endpoints_intersections(skel01: np.ndarray):
    """Return lists of (x,y) endpoints and intersections from a 0/1 skeleton."""
    H, W = skel01.shape
    endpoints = []
    intersections = []
    for y in range(1, H - 1):
        for x in range(1, W - 1):
            if skel01[y, x] != 1:
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
    Q3, Q4, Q6, _dummy_q8, sigma, g_mean, N100, _dummy_nv = calculate_q3(
        R_mask, gray, Sunocc, include_q8=False
    )

    # --- Q5
    Q5, H_bits = calculate_q5(R_mask, gray, bit_depth=8, ep_c=0.75)

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

    # Determine Q8/Q9 parameters based on the method (MC vs others)
    method = os.path.basename(os.path.normpath(vein_root)).upper()

    if method == "MC":
        params = dict(min_area=30, prune_iters=8, min_skel_len=0, keep_largest=False)
    else:
        params = dict(min_area=120, prune_iters=30, min_skel_len=120, keep_largest=True)
    # --- Q8 (ISO): total vascular length
    # Finger second phalanx -> Lc = 600 (Table 2 in the draft)
    Q8, N_vessel, skel_q8 = calculate_q8(R_mask, vein_img, Lc=600, **params)
    # --- Q9 (ISO): number of feature points
    # Finger second phalanx -> Fc = 15 (Table 3 in the draft)
    Q9, N_fp, N_end, N_int, skel_q9 = calculate_q9(R_mask, vein_img, Fc=15, **params)
    
    
    # Q8, N_vessel, skel_q8 = calculate_q8(R_mask, vein_img, Lc=600)
    # Q9, N_fp, N_end, N_int, skel_q9 = calculate_q9(R_mask, vein_img, Fc=15)
    
    # For visualization/debugging: extract endpoint and intersection coordinates from the skeleton
    end_pts, int_pts = get_endpoints_intersections(skel_q9)
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
        "vein_map": vein_path,
        "debug_dir": out_dir
    }
    return result


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to an image OR a folder of images")
    ap.add_argument("--vein_root", required=True, help="Root folder containing MATLAB vein maps (e.g., debug_openvein_features)")
    ap.add_argument("--out", default="debug_outputs", help="Where to save debug images")
    args = ap.parse_args()

    if os.path.isdir(args.input):
        paths = sorted(glob.glob(os.path.join(args.input, "*.*")))
    else:
        paths = [args.input]

    ensure_dir(args.out)

    all_results = []
    headers = ["image", "Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8", "Q9", "N_vessel", "N_end", "N_int", "N_fp"]
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
            r["N_vessel"], r["N_end"], r["N_int"], r["N_fp"]
        ])

    print("\nQUALITY SCORES (ISO/IEC 29794-9)\n")
    print(tabulate(table, headers=headers, tablefmt="github"))

if __name__ == "__main__":
    main()
