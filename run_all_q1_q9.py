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
from q8 import calculate_q8
from q9 import calculate_q9


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def overlay_mask(gray: np.ndarray, mask255: np.ndarray) -> np.ndarray:
    """Return a 3-channel visualization: gray + green foreground overlay."""
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    fg = (mask255 == 255)
    vis[fg] = (0.6 * vis[fg] + 0.4 * np.array([0, 255, 0])).astype(np.uint8)
    return vis


# def simple_vessel_extraction(gray: np.ndarray, R_mask255: np.ndarray) -> np.ndarray:
#     """
#     Simple vessel extraction to make Q8/Q9 testable:
#     - restrict to foreground
#     - CLAHE (contrast)
#     - black-hat morphology (dark lines)
#     - Otsu threshold INV
#     Output: vessel_binary uint8 {0,255}
#     """
#     fg = (R_mask255 == 255)
#     if np.count_nonzero(fg) == 0:
#         return np.zeros_like(gray, dtype=np.uint8)

#     # Apply mask
#     roi = gray.copy()
#     roi[~fg] = 0

#     # Contrast enhance (CLAHE)
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     roi_c = clahe.apply(roi)

#     # Black-hat to highlight dark vessels on brighter tissue
#     k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
#     bh = cv2.morphologyEx(roi_c, cv2.MORPH_BLACKHAT, k)

#     # Threshold (vessels usually become bright in black-hat output)
#     # Use Otsu on foreground-only pixels
#     vals = bh[fg]
#     if vals.size == 0:
#         return np.zeros_like(gray, dtype=np.uint8)

#     # Otsu threshold on a 1D array safely via numpy percentile fallback:
#     # We’ll just compute Otsu on the full masked image (works fine because background=0).
#     _, vessel = cv2.threshold(bh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

#     # Clean up small noise
#     vessel = cv2.morphologyEx(vessel, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
#     vessel[~fg] = 0
#     return vessel


# def mark_points_on_skeleton(skel01: np.ndarray, endpoints: list[tuple[int,int]], intersections: list[tuple[int,int]]) -> np.ndarray:
#     """Visualize skeleton + endpoints/intersections."""
#     vis = (skel01.astype(np.uint8) * 255)
#     vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

#     for (y, x) in endpoints:
#         cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)  # green
#     for (y, x) in intersections:
#         cv2.circle(vis, (x, y), 2, (0, 0, 255), -1)  # red
#     return vis


# def extract_points_from_q9(vessel_binary: np.ndarray, fg_mask_bool: np.ndarray):
#     """
#     Recompute skeleton + find endpoints/intersections for visualization.
#     Uses the same transition-number idea as your Q9.
#     """
#     import cv2
#     from q9 import skeletonize_binary, transition_number_8

#     vb = np.zeros_like(vessel_binary, dtype=np.uint8)
#     vb[fg_mask_bool] = (vessel_binary[fg_mask_bool] > 0).astype(np.uint8) * 255
#     skel = skeletonize_binary(vb)  # 0/1
#     skel = skel * fg_mask_bool.astype(np.uint8)

#     padded = np.pad(skel, 1, mode="constant", constant_values=0)
#     ys, xs = np.where(skel == 1)

#     endpoints = []
#     intersections = []

#     for y, x in zip(ys, xs):
#         yy, xx = y + 1, x + 1
#         block = padded[yy-1:yy+2, xx-1:xx+2]
#         neigh = np.array([
#             block[0, 1],  # p2
#             block[0, 2],  # p3
#             block[1, 2],  # p4
#             block[2, 2],  # p5
#             block[2, 1],  # p6
#             block[2, 0],  # p7
#             block[1, 0],  # p8
#             block[0, 0],  # p9
#         ], dtype=np.uint8)

#         T = transition_number_8(neigh)
#         if T == 2:
#             endpoints.append((y, x))
#         elif T == 6 or T == 8:
#             intersections.append((y, x))

#     return skel, endpoints, intersections


def run_on_image(image_path: str, out_dir: str):
    ensure_dir(out_dir)
    base = os.path.splitext(os.path.basename(image_path))[0]

    # --- Q1
    Q1, Sunocc, R_mask, gray = calculate_q1(image_path)

    # --- Q2 (your function re-segments; OK for now)
    # Q2, cx, cy, S_H, S_V = calculate_q2(image_path)
    Q2, cx, cy, S_H, S_V = calculate_q2(R_mask, gray)

    # --- Q3 + Q4 + Q6 + (disable q8 inside q3 so we do q8 with debug)
    Q3, Q4, Q6, _dummy_q8, sigma, g_mean, N100, _dummy_nv = calculate_q3(R_mask, gray, Sunocc, include_q8=False)

    # --- Q5
    # Q5, H_bits = calculate_q5(R_mask, gray)
    Q5, H_bits = calculate_q5(R_mask, gray, bit_depth=8, ep_c=0.75)

    # --- Q7
    # Q7, block_var = calculate_q7(R_mask, gray, g_mean)
    Q7, block_var = calculate_q7(R_mask, gray, g_mean)
    
    # --- Q8 (ISO-aligned with debug outputs)
    Q8, N_vessel, vessel_binary_q8, skeleton_q8 = calculate_q8(R_mask, gray, Lc=600, return_debug=True)


    # --- Vessel extraction for Q9 (test pipeline)
    fg_bool = (R_mask == 255)
    # vessel_binary = simple_vessel_extraction(gray, R_mask)

    # --- Q9
    Q9, Ne, Ni = calculate_q9(vessel_binary_q8,
    fg_bool,
    capture_site="entire",        # PALMAR dataset
    includes_fingers=False,     # set True only if ROI contains fingers
    min_skel_component_len=8)

    # --- Save debug images
    cv2.imwrite(os.path.join(out_dir, f"{base}_gray.png"), gray)
    cv2.imwrite(os.path.join(out_dir, f"{base}_Rmask.png"), R_mask)
    cv2.imwrite(os.path.join(out_dir, f"{base}_overlay_R.png"), overlay_mask(gray, R_mask))
    # cv2.imwrite(os.path.join(out_dir, f"{base}_vessel_binary.png"), vessel_binary)
    cv2.imwrite(os.path.join(out_dir, f"{base}_q8_vessel_binary.png"), vessel_binary_q8)
    cv2.imwrite(os.path.join(out_dir, f"{base}_q8_skeleton.png"), skeleton_q8)


    # skel, endpoints, intersections = extract_points_from_q9(vessel_binary_q8, fg_bool)
    # cv2.imwrite(os.path.join(out_dir, f"{base}_skeleton.png"), skel.astype(np.uint8) * 255)
    # cv2.imwrite(os.path.join(out_dir, f"{base}_q9_points.png"),
    #             mark_points_on_skeleton(skel, endpoints, intersections))

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
        "Q9": Q9, "Ne": int(Ne), "Ni": int(Ni),
        "debug_dir": out_dir
    }
    return result


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to an image OR a folder of images")
    ap.add_argument("--out", default="debug_outputs", help="Where to save debug images")
    args = ap.parse_args()

    if os.path.isdir(args.input):
        paths = sorted(glob.glob(os.path.join(args.input, "*.*")))
    else:
        paths = [args.input]

    ensure_dir(args.out)
    
    all_results = []
    # After processing all images
    headers = ["image", "Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8", "Q9","Ne","Ni"]
    for p in paths:
        try:
            r = run_on_image(p, args.out)
            all_results.append(r)
            
        except Exception as e:
            print(f"ERROR on {p}: {e}")

    table = []
    for r in all_results:
        table.append([
            r["file"],
            r["Q1"], r["Q2"], r["Q3"], r["Q4"], r["Q5"],
            r["Q6"], r["Q7"], r["Q8"], r["Q9"],r['Ne'],r['Ni']
        ])

    print("\nQUALITY SCORES (ISO/IEC 29794-9)\n")
    print(tabulate(table, headers=headers, tablefmt="github"))


    # Save CSV summary
    import csv
    csv_path = os.path.join(args.out, "summary_q1_q9.csv")
    keys = ["file","Q1","Q2","Q3","Q4","Q5","Q6","Q7","Q8","Q9","Ne","Ni"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in all_results:
            w.writerow({k: r.get(k, "") for k in keys})

    print(f"\nSaved debug images + summary CSV to: {args.out}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
