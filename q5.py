"""
ISO/IEC 29794-9 Quality Component 5 (Information Entropy)
Finger Vascular Biometrics - Second Phalanx

Definition (Clause 5.2.5 style):
- Compute Shannon entropy H of grayscale values **inside the foreground region R**.
- Normalize H to [0, 100] using the bit depth (D) and an ISO scaling factor.
- Veto: if R is invalid (area = 0), Q5 = 0.

Score (integer 0..100):
    H_bits = - Σ p_k * log2(p_k), over gray levels k present in R
    Q5 = ROUND( min(100, max(0, (H_bits / D) * 100 * scale)) )

Notes:
- D = bit depth (8 by default) → max entropy = D bits.
- `scale` (default 0.75) is included to match common draft scaling practice.
"""

import numpy as np
from typing import Tuple

def _shannon_entropy(values: np.ndarray, bit_depth: int = 8) -> float:
    """Shannon entropy (bits) over integer grayscale values."""
    if values.size == 0:
        return 0.0
    # Build histogram only over present bins for numeric stability
    L = 2 ** bit_depth
    hist = np.bincount(values.astype(np.uint8), minlength=L).astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 0.0
    p = hist / total
    # Avoid log2(0): only nonzero probabilities contribute
    nz = p > 0
    H = -np.sum(p[nz] * np.log2(p[nz]))
    return float(H)

def calculate_q5(R_mask: np.ndarray,
                 Grayscale_Image: np.ndarray,
                 bit_depth: int = 8,
                 scale: float = 0.75) -> Tuple[int, float]:
    """
    Q5 — Information Entropy (inside foreground region R).

    Args:
        R_mask: binary mask, 255=foreground, 0=background  (from Q1)
        Grayscale_Image: original grayscale image
        bit_depth: image bit depth (default 8)
        scale: optional ISO scaling factor (default 0.75)

    Returns:
        (Q5_score, H_bits)
    """
    # Veto: invalid or empty foreground region
    fg = (R_mask == 255)
    N = int(np.sum(fg))
    if N == 0:
        return 0, 0.0

    # Collect pixel values within R (uint8 expected for 8-bit images)
    vals = Grayscale_Image[fg].astype(np.uint8)

    # Shannon entropy in bits
    H_bits = _shannon_entropy(vals, bit_depth=bit_depth)

    # Normalize to [0, 100] and apply optional ISO scale
    q5_raw = (H_bits / float(bit_depth)) * 100.0 * float(scale)
    Q5 = int(round(q5_raw))
    Q5 = max(0, min(100, Q5))
    return Q5, H_bits



# ******************************ISO-aligned*************************************

def calculate_q5_ISO(
    R_mask: np.ndarray,
    Grayscale_Image: np.ndarray,
    bit_depth: int = 8,
    ep_c: float = 0.75
) -> Tuple[int, float]:
    """
    Q5 — Information Entropy (ISO/IEC 29794-9 PWI draft, Formula (10)).

    Steps:
    1) Compute Shannon entropy H_bits inside foreground region R.
    2) Normalize: ep = H_bits / D
    3) Map to [0,100] using ep_c=0.75:
          Q5 = min(100, round((ep / ep_c) * 100))

    Veto:
    - If R is invalid (area=0), Q5 = 0

    Returns:
        (Q5_score, H_bits)
    """
    fg = (R_mask == 255)
    if np.count_nonzero(fg) == 0:
        return 0, 0.0

    vals = Grayscale_Image[fg].astype(np.uint8)
    H_bits = _shannon_entropy(vals, bit_depth=bit_depth)

    # Normalized entropy ep in [0,1] (for D-bit images)
    ep = H_bits / float(bit_depth) if bit_depth > 0 else 0.0

    # ISO/PWI Formula (10): Q5 = min(100, round((ep / ep_c) * 100))
    if ep_c <= 0:
        return 0, H_bits

    q5_raw = (ep / float(ep_c)) * 100.0
    Q5 = int(round(q5_raw))
    Q5 = max(0, min(100, Q5))

    return Q5, H_bits
