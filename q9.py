import numpy as np
import cv2

def skeletonize_binary(binary_img: np.ndarray) -> np.ndarray:
    """
    Skeletonize a binary image (uint8 0/255 or 0/1) into a 1-pixel-wide skeleton.
    Returns uint8 image with values 0/1.
    """
    # Normalize to 0/1
    b = (binary_img > 0).astype(np.uint8)

    # Try OpenCV contrib thinning first (fast + good)
    try:
        skel = cv2.ximgproc.thinning(b * 255, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
        return (skel > 0).astype(np.uint8)
    except Exception:
        pass

    # Fallback: scikit-image skeletonize
    try:
        from skimage.morphology import skeletonize
        skel = skeletonize(b.astype(bool))
        return skel.astype(np.uint8)
    except Exception as e:
        raise RuntimeError(
            "Skeletonization failed. Install either opencv-contrib-python "
            "or scikit-image.\n"
            "pip install opencv-contrib-python\n"
            "or\n"
            "pip install scikit-image"
        ) from e


def transition_number_8(neigh: np.ndarray) -> int:
    """
    neigh: 8 neighbors in this order around center pixel:
      p2 p3 p4
      p9  c p5
      p8 p7 p6

    We compute the number of 0->1 transitions along the circular sequence:
    p2 -> p3 -> p4 -> p5 -> p6 -> p7 -> p8 -> p9 -> p2
    """
    # Ensure 0/1
    n = (neigh > 0).astype(np.uint8)

    # Circular sequence
    seq = np.array([n[0], n[1], n[2], n[3], n[4], n[5], n[6], n[7], n[0]], dtype=np.uint8)

    # Count 0->1 transitions
    return int(np.sum((seq[:-1] == 0) & (seq[1:] == 1)))


def compute_q9_feature_points(
    vessel_binary: np.ndarray,
    foreground_mask: np.ndarray,
    Cf: int = 15
) -> tuple[int, int, int]:
    """
    Compute Q9 and return (Q9_score, Ne_endpoints, Ni_intersections).

    vessel_binary: uint8 or bool, vessels=1/255, background=0
    foreground_mask: bool mask defining valid ROI (R_mask)
    Cf: normalization constant (finger vein, 2nd phalanx -> 15)
    """
    if vessel_binary.shape != foreground_mask.shape:
        raise ValueError("vessel_binary and foreground_mask must have the same shape")

    # Apply mask
    vb = np.zeros_like(vessel_binary, dtype=np.uint8)
    vb[foreground_mask] = (vessel_binary[foreground_mask] > 0).astype(np.uint8) * 255

    # Skeletonize
    skel = skeletonize_binary(vb)  # 0/1

    # Only count skeleton inside foreground
    skel = skel * foreground_mask.astype(np.uint8)

    h, w = skel.shape
    Ne = 0  # endpoints
    Ni = 0  # intersections

    # Pad so we can safely take 3x3 neighborhoods
    padded = np.pad(skel, pad_width=1, mode="constant", constant_values=0)

    # Iterate over skeleton pixels
    ys, xs = np.where(skel == 1)
    for y, x in zip(ys, xs):
        # 3x3 neighborhood around (y,x) in padded coordinates
        yy, xx = y + 1, x + 1
        block = padded[yy-1:yy+2, xx-1:xx+2]

        # Neighbors in ISO-style ring order (p2..p9):
        # p2=top, p3=top-right, p4=right, p5=bottom-right,
        # p6=bottom, p7=bottom-left, p8=left, p9=top-left
        neigh = np.array([
            block[0, 1],  # p2
            block[0, 2],  # p3
            block[1, 2],  # p4
            block[2, 2],  # p5
            block[2, 1],  # p6
            block[2, 0],  # p7
            block[1, 0],  # p8
            block[0, 0],  # p9
        ], dtype=np.uint8)

        T = transition_number_8(neigh)

        # ISO rule:
        # endpoint: T == 2
        # intersection: T == 6 or T == 8
        if T == 2:
            Ne += 1
        elif T == 6 or T == 8:
            Ni += 1

    # Q9 score
    q9 = int(min(100, round(((Ne + Ni) / float(Cf)) * 100)))

    return q9, Ne, Ni


# ---------------- Example usage ----------------
if __name__ == "__main__":
    # Example placeholders:
    # gray = cv2.imread("your_image.png", cv2.IMREAD_GRAYSCALE)
    # foreground_mask = ... (bool)
    # vessel_binary = ... (uint8 0/255)

    # q9, Ne, Ni = compute_q9_feature_points(vessel_binary, foreground_mask, Cf=15)
    # print("Q9:", q9, "Ne:", Ne, "Ni:", Ni)
    pass
