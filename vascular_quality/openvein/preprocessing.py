"""
OpenVein-style preprocessing chain (Python adapters).

MATLAB reference (OpenVein Matcher.preprocessImageSet):
  ToDouble, LeeRegion, Zhao09, CLAHE, Zhang09
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

import cv2
import numpy as np

Array = np.ndarray


class PreprocessParity(str, Enum):
    OPENVEIN = "openvein"
    APPROXIMATE = "approximate"
    UNAVAILABLE = "unavailable"


class PreprocessingStepUnavailable(NotImplementedError):
    """Raised when an OpenVein preprocessing step has no Python implementation yet."""


@dataclass(frozen=True)
class PreprocessStepSpec:
    name: str
    openvein_name: str
    parity: PreprocessParity
    apply: Callable[[Array], Array]
    description: str


def to_double(image: Array) -> Array:
    """OpenVein ToDouble: grayscale to float64 in [0, 1]."""
    gray = image.astype(np.float64)
    if gray.max() > 1.0:
        gray = gray / 255.0
    return np.clip(gray, 0.0, 1.0)


def lee_region(image: Array) -> Array:
    """
    Finger-region mask (approximate LeeRegion).

    TODO: Port exact OpenVein LeeRegion parameters and morphology.
    Current: Otsu foreground + largest component, masked float image.
    """
    src = image
    if src.dtype != np.uint8:
        src = (np.clip(src, 0.0, 1.0) * 255).astype(np.uint8)

    _, mask = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    mask = _largest_component(mask)

    out = src.astype(np.float64) / 255.0
    out[mask == 0] = 0.0
    return out


def zhao09(image: Array) -> Array:
    """
    Zhao et al. ROI normalization (OpenVein Zhao09).

    TODO: Port OpenVein Zhao09 — not yet available in Python.
  """
    raise PreprocessingStepUnavailable(
        "Zhao09 preprocessing is not implemented. "
        "OpenVein MATLAB uses Zhao09 between LeeRegion and CLAHE. "
        "Use --skip-unavailable-preprocess to run without this step."
    )


def clahe(image: Array) -> Array:
    """CLAHE contrast enhancement (OpenVein CLAHE equivalent via OpenCV)."""
    src = image
    if src.dtype != np.uint8:
        src = (np.clip(src, 0.0, 1.0) * 255).astype(np.uint8)

    clahe_op = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe_op.apply(src)
    return enhanced.astype(np.float64) / 255.0


def zhang09(image: Array) -> Array:
    """
    Zhang et al. normalization (OpenVein Zhang09).

    TODO: Port OpenVein Zhang09 — not yet available in Python.
    """
    raise PreprocessingStepUnavailable(
        "Zhang09 preprocessing is not implemented. "
        "OpenVein MATLAB uses Zhang09 after CLAHE. "
        "Use --skip-unavailable-preprocess to run without this step."
    )


def _largest_component(mask: Array) -> Array:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8
    )
    if num_labels <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = 1 + int(np.argmax(areas))
    return np.where(labels == largest, 255, 0).astype(np.uint8)


DEFAULT_PREPROCESS_CHAIN: tuple[str, ...] = (
    "ToDouble",
    "LeeRegion",
    "Zhao09",
    "CLAHE",
    "Zhang09",
)

PREPROCESS_STEPS: dict[str, PreprocessStepSpec] = {
    "ToDouble": PreprocessStepSpec(
        name="ToDouble",
        openvein_name="ToDouble",
        parity=PreprocessParity.OPENVEIN,
        apply=to_double,
        description="Convert to float64 [0, 1].",
    ),
    "LeeRegion": PreprocessStepSpec(
        name="LeeRegion",
        openvein_name="LeeRegion",
        parity=PreprocessParity.APPROXIMATE,
        apply=lee_region,
        description="Finger ROI mask (approximate; not OpenVein-identical).",
    ),
    "Zhao09": PreprocessStepSpec(
        name="Zhao09",
        openvein_name="Zhao09",
        parity=PreprocessParity.UNAVAILABLE,
        apply=zhao09,
        description="Zhao ROI normalization — placeholder.",
    ),
    "CLAHE": PreprocessStepSpec(
        name="CLAHE",
        openvein_name="CLAHE",
        parity=PreprocessParity.APPROXIMATE,
        apply=clahe,
        description="CLAHE via OpenCV (parameters may differ from OpenVein).",
    ),
    "Zhang09": PreprocessStepSpec(
        name="Zhang09",
        openvein_name="Zhang09",
        parity=PreprocessParity.UNAVAILABLE,
        apply=zhang09,
        description="Zhang normalization — placeholder.",
    ),
}


def preprocess_openvein(
    image: Array,
    *,
    steps: tuple[str, ...] | None = None,
    skip_unavailable: bool = False,
) -> Array:
    """
    Run the OpenVein preprocessing chain on a single grayscale image.

    Args:
        image: Grayscale uint8 or float array.
        steps: Ordered step names (default: full OpenVein chain).
        skip_unavailable: If True, skip UNAVAILABLE steps with a printed notice.
    """
    chain = steps or DEFAULT_PREPROCESS_CHAIN
    out = image
    for step_name in chain:
        spec = PREPROCESS_STEPS.get(step_name)
        if spec is None:
            raise ValueError(
                f"Unknown preprocessing step {step_name!r}. "
                f"Expected one of: {', '.join(PREPROCESS_STEPS)}."
            )
        if spec.parity == PreprocessParity.UNAVAILABLE and skip_unavailable:
            print(f"  [skip] {spec.openvein_name}: {spec.description}")
            continue
        out = spec.apply(out)
    return out
