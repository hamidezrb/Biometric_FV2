"""Generate Q1 Clause 5.2.1 verification masks and numeric breakdown for one image."""

from __future__ import annotations

import argparse
import os

import cv2
import numpy as np

from iso_constants import CaptureSite
from q1 import calculate_q1_detailed


def verify_q1_image(
    image_path: str,
    out_dir: str,
    capture_site: CaptureSite = CaptureSite.FINGER_SECOND_PHALANX,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]

    result = calculate_q1_detailed(image_path, capture_site=capture_site)
    gray = result.grayscale
    fg = result.R_foreground
    occ = result.occlusion_mask
    unocc = result.R_unoccluded

    cv2.imwrite(os.path.join(out_dir, f"{base}_foreground_mask.png"), fg)
    cv2.imwrite(os.path.join(out_dir, f"{base}_occlusion_or_saturation_mask.png"), occ)
    cv2.imwrite(os.path.join(out_dir, f"{base}_unoccluded_effective_area_mask.png"), unocc)

    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    u = unocc == 255
    o = (occ == 255) & (fg == 255)
    vis[u] = (0.45 * vis[u] + 0.55 * np.array([0, 255, 0])).astype(np.uint8)
    vis[o] = (0.35 * vis[o] + 0.65 * np.array([0, 0, 255])).astype(np.uint8)
    cv2.imwrite(
        os.path.join(out_dir, f"{base}_overlay_unoccluded_green_occluded_red.png"),
        vis,
    )

    non255_in_occ = int(np.count_nonzero((occ == 255) & (gray < 255)))
    if unocc.any():
        umin, umax = int(gray[u].min()), int(gray[u].max())
    else:
        umin = umax = 0

    print(f"=== Q1 verification: {base} ===")
    print(f"foreground pixel count:                 {result.S_foreground}")
    print(f"occluded/saturated foreground pixels:   {result.S_occluded}")
    print(f"unoccluded effective pixel count A:     {result.S_unoccluded}")
    print(f"coefficient c (Sc):                     {result.Sc}")
    print(f"raw score before rounding:              {result.q1_raw:.4f}")
    print(f"final Q1:                               {result.Q1_score}")
    print()
    print(f"occlusion pixels with intensity < 255:  {non255_in_occ}")
    print(f"unoccluded intensity range:             {umin} - {umax}")
    print(f"outputs: {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", default="debug_outputs_q1_verify")
    parser.add_argument(
        "--capture-site",
        default=CaptureSite.FINGER_SECOND_PHALANX.value,
        choices=[site.value for site in CaptureSite],
    )
    args = parser.parse_args()
    verify_q1_image(args.input, args.out, CaptureSite(args.capture_site))


if __name__ == "__main__":
    main()
