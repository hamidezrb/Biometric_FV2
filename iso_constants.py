"""
ISO/IEC 29794-9 (PWI draft) capture-site coefficients — Tables 1, 2, and 3.

Effective Area (SC), Total Vascular Length (LC), Feature Points (FC).
"""

from __future__ import annotations

from enum import Enum
from typing import TypedDict


class CaptureSite(Enum):
    """Capture-site variants from ISO/IEC 29794-9 Tables 1–3."""

    FINGER_SECOND_PHALANX = "finger_second_phalanx"
    PALM_OR_DORSAL = "palm_or_dorsal"
    FULL_HAND = "full_hand"


class CaptureSiteCoefficients(TypedDict):
    SC: int
    LC: int
    FC: int


# Table 1 (SC), Table 2 (LC), Table 3 (FC) — PWI-draft-29794-9-250924
CAPTURE_SITE_COEFFICIENTS: dict[CaptureSite, CaptureSiteCoefficients] = {
    CaptureSite.FINGER_SECOND_PHALANX: {
        "SC": 20000,
        "LC": 600,
        "FC": 15,
    },
    CaptureSite.PALM_OR_DORSAL: {
        "SC": 40000,
        "LC": 1200,
        "FC": 25,
    },
    CaptureSite.FULL_HAND: {
        "SC": 300000,
        "LC": 9000,
        "FC": 50,
    },
}

DEFAULT_CAPTURE_SITE = CaptureSite.FINGER_SECOND_PHALANX

# Non capture-site-specific ISO constants (unchanged across sites in draft)
GC_SHARPNESS = 0.006
EP_C_ENTROPY = 0.75
Q7_BLOCK_NORMALIZER = 121


def get_capture_site_coefficients(
    capture_site: CaptureSite = DEFAULT_CAPTURE_SITE,
) -> CaptureSiteCoefficients:
    """Return SC, LC, and FC for the given capture site."""
    return CAPTURE_SITE_COEFFICIENTS[capture_site]


# Backward-compatible aliases (finger second phalanx defaults)
SC_FINGER_SECOND_PHALANX = CAPTURE_SITE_COEFFICIENTS[CaptureSite.FINGER_SECOND_PHALANX]["SC"]
LC_FINGER_SECOND_PHALANX = CAPTURE_SITE_COEFFICIENTS[CaptureSite.FINGER_SECOND_PHALANX]["LC"]
FC_FINGER_SECOND_PHALANX = CAPTURE_SITE_COEFFICIENTS[CaptureSite.FINGER_SECOND_PHALANX]["FC"]
