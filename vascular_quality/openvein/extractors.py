"""
OpenVein feature extractors (Python adapters).

Each extractor exposes parity status so callers know whether output matches
the OpenVein MATLAB toolkit or is still a TODO placeholder.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

import cv2
import numpy as np

Array = np.ndarray


class ExtractorParity(str, Enum):
    OPENVEIN = "openvein"
    APPROXIMATE = "approximate"
    UNAVAILABLE = "unavailable"


class ExtractorUnavailable(NotImplementedError):
    """Raised when an extractor has no Python implementation yet."""


@dataclass(frozen=True)
class ExtractorSpec:
    tag: str
    full_name: str
    openvein_feature_type: str
    parity: ExtractorParity
    extract: Callable[[Array], Array]
    notes: str = ""


def _to_uint8_working(gray: Array) -> Array:
    if gray.dtype == np.uint8:
        return gray
    if gray.max() <= 1.0:
        return (np.clip(gray, 0.0, 1.0) * 255).astype(np.uint8)
    return np.clip(gray, 0, 255).astype(np.uint8)


def _binarize_vein_response(response: Array, percentile: float = 85.0) -> Array:
    """Threshold enhanced response into a binary vein map."""
    flat = response[response > 0]
    if flat.size == 0:
        return np.zeros_like(response, dtype=np.uint8)
    thresh = float(np.percentile(flat, percentile))
    return (response >= thresh).astype(np.uint8)


def extract_rlt(image: Array) -> Array:
    """
    Repeated Line Tracking (RLT).

    TODO: Port OpenVein FeatureType.RepeatedLineTracking exactly.
    Current: oriented dark-line enhancement (approximate only).
    """
    gray = _to_uint8_working(image)
    acc = np.zeros_like(gray, dtype=np.float64)
    for angle in range(0, 180, 10):
        ksize = 15
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, 3))
        rotated = _rotate_kernel(kernel, angle)
        opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, rotated)
        acc = np.maximum(acc, (gray.astype(np.float64) - opened.astype(np.float64)))
    acc = np.clip(acc, 0, None)
    return _binarize_vein_response(acc)


def extract_mc(image: Array) -> Array:
    """
    Maximum Curvature (MC).

    TODO: Port OpenVein FeatureType.MaximumCurvature.
    """
    raise ExtractorUnavailable(
        "MC (Maximum Curvature) is not implemented in Python yet. "
        "Requires porting OpenVein FeatureType.MaximumCurvature."
    )


def extract_wld(image: Array) -> Array:
    """
    Huang Wide Line Detector (WLD).

    TODO: Port OpenVein FeatureType.HuangWideLine.
    """
    raise ExtractorUnavailable(
        "WLD (Huang Wide Line) is not implemented in Python yet. "
        "Requires porting OpenVein FeatureType.HuangWideLine."
    )


def extract_pc(image: Array) -> Array:
    """
    Principal Curvature (PC).

    TODO: Port OpenVein FeatureType.PrincipalCurvature.
    """
    raise ExtractorUnavailable(
        "PC (Principal Curvature) is not implemented in Python yet. "
        "Requires porting OpenVein FeatureType.PrincipalCurvature."
    )


def extract_gf(image: Array) -> Array:
    """
    Kumar Gabor filter bank (GF).

    TODO: Match OpenVein FeatureType.KumarGabor parameters exactly.
    Current: multi-orientation Gabor magnitude (approximate).
    """
    gray = _to_uint8_working(image).astype(np.float32)
    responses: list[Array] = []
    for theta in np.linspace(0, np.pi, 12, endpoint=False):
        kernel = cv2.getGaborKernel(
            ksize=(21, 21),
            sigma=4.0,
            theta=float(theta),
            lambd=10.0,
            gamma=0.5,
            psi=0,
        )
        filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
        responses.append(np.abs(filtered))
    combined = np.max(np.stack(responses, axis=0), axis=0)
    return _binarize_vein_response(combined)


def extract_emc(image: Array) -> Array:
    """
    EMC extractor.

    TODO: Port OpenVein FeatureType.EMC.
    """
    raise ExtractorUnavailable(
        "EMC is not implemented in Python yet. "
        "Requires porting OpenVein FeatureType.EMC."
    )


def _rotate_kernel(kernel: Array, angle_deg: float) -> Array:
    h, w = kernel.shape
    center = (w / 2, h / 2)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(
        kernel.astype(np.float32),
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return (rotated > 0.5).astype(np.uint8)


EXTRACTORS: dict[str, ExtractorSpec] = {
    "RLT": ExtractorSpec(
        tag="RLT",
        full_name="Repeated Line Tracking",
        openvein_feature_type="FeatureType.RepeatedLineTracking",
        parity=ExtractorParity.APPROXIMATE,
        extract=extract_rlt,
        notes="Oriented morphology approximation; not OpenVein-identical.",
    ),
    "MC": ExtractorSpec(
        tag="MC",
        full_name="Maximum Curvature",
        openvein_feature_type="FeatureType.MaximumCurvature",
        parity=ExtractorParity.UNAVAILABLE,
        extract=extract_mc,
    ),
    "WLD": ExtractorSpec(
        tag="WLD",
        full_name="Huang Wide Line",
        openvein_feature_type="FeatureType.HuangWideLine",
        parity=ExtractorParity.UNAVAILABLE,
        extract=extract_wld,
    ),
    "PC": ExtractorSpec(
        tag="PC",
        full_name="Principal Curvature",
        openvein_feature_type="FeatureType.PrincipalCurvature",
        parity=ExtractorParity.UNAVAILABLE,
        extract=extract_pc,
    ),
    "GF": ExtractorSpec(
        tag="GF",
        full_name="Kumar Gabor",
        openvein_feature_type="FeatureType.KumarGabor",
        parity=ExtractorParity.APPROXIMATE,
        extract=extract_gf,
        notes="Gabor bank approximation; parameters differ from OpenVein.",
    ),
    "EMC": ExtractorSpec(
        tag="EMC",
        full_name="EMC",
        openvein_feature_type="FeatureType.EMC",
        parity=ExtractorParity.UNAVAILABLE,
        extract=extract_emc,
    ),
}

EXTRACTOR_NAMES: tuple[str, ...] = tuple(EXTRACTORS.keys())


def get_extractor(tag: str) -> ExtractorSpec:
    key = tag.upper()
    if key not in EXTRACTORS:
        raise ValueError(
            f"Unknown extractor {tag!r}. Expected one of: {', '.join(EXTRACTOR_NAMES)}."
        )
    return EXTRACTORS[key]


def list_extractors() -> list[ExtractorSpec]:
    return [EXTRACTORS[k] for k in EXTRACTOR_NAMES]


def extract_feature(image: Array, extractor: str) -> Array:
    """Run one extractor on a preprocessed grayscale image."""
    spec = get_extractor(extractor)
    if spec.parity == ExtractorParity.UNAVAILABLE:
        spec.extract(image)  # always raises ExtractorUnavailable
    return spec.extract(image)
