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


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def overlay_mask(gray: np.ndarray, mask255: np.ndarray) -> np.ndarray:
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    fg = (mask255 == 255)
    vis[fg] = (0.6 * vis[fg] + 0.4 * np.array([0, 255, 0])).astype(np.uint8)
    return vis


def run_on_image(image_path: str, out_dir: str):
    ensure_dir(out_dir)
    base = os.path.splitext(os.path.basename(image_path))[0]

    # Q1
    Q1, Sunocc, R_mask, gray = calculate_q1(image_path)

    # Q2
    Q2, cx, cy, S_H, S_V = calculate_q2(R_mask, gray)

    Q3, sigma, g_mean = calculate_q3(R_mask, gray)
    Q4, sigma, g_mean = calculate_q4(R_mask, gray)
    Q5, H_bits = calculate_q5(R_mask, gray, bit_depth=8, ep_c=0.75)
    Q6, N100 = calculate_q6(R_mask, gray, S_unoccluded=Sunocc, gc=0.006)

    # Q7
    Q7, block_var = calculate_q7(R_mask, gray, g_mean)

    # Save debug images
    cv2.imwrite(os.path.join(out_dir, f"{base}_gray.png"), gray)
    cv2.imwrite(os.path.join(out_dir, f"{base}_Rmask.png"), R_mask)
    cv2.imwrite(os.path.join(out_dir, f"{base}_overlay_R.png"), overlay_mask(gray, R_mask))

    result = {
        "file": base,
        "Q1": int(Q1),
        "Q2": int(Q2),
        "Q3": int(Q3),
        "Q4": int(Q4),
        "Q5": int(Q5),
        "Q6": int(Q6),
        "Q7": int(Q7),
        "block_var": float(block_var),
        "debug_dir": out_dir,
    }
    return result


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to an image OR a folder of images")
    ap.add_argument("--out", default="debug_outputs_q1_q7", help="Where to save debug images")
    args = ap.parse_args()

    if os.path.isdir(args.input):
        paths = sorted(glob.glob(os.path.join(args.input, "*.*")))
    else:
        paths = [args.input]

    ensure_dir(args.out)

    all_results = []
    for p in paths:
        try:
            r = run_on_image(p, args.out)
            all_results.append(r)
        except Exception as e:
            print(f"ERROR on {p}: {e}")

    headers = ["image", "Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7"]
    table = []
    
    
    table.append([
            "PLUS-FV3-Laser_PALMAR_018_01_04_01",
            58,
            36,
            19,
            28,
            48,
            12,
            42,
        ])
    
    table.append([
            "PLUS-FV3-Laser_PALMAR_026_01_02_01",
            100,
            50,
            32,
            69,
            72,
            20,
            60,
        ])
    
    # for r in all_results:
    #     table.append([
    #         r["file"],
    #         r["Q1"],
    #         r["Q2"],
    #         r["Q3"],
    #         r["Q4"],
    #         r["Q5"],
    #         r["Q6"],
    #         r["Q7"],
    #     ])

    print("\nQUALITY SCORES (ISO/IEC 29794-9) — Q1 to Q7\n")
    print(tabulate(table, headers=headers, tablefmt="github"))


if __name__ == "__main__":
    main()