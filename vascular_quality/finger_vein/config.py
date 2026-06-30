"""Finger-vein dataset configuration (PLUS, IDIAP, SCUT)."""

from __future__ import annotations

from iso_constants import CaptureSite
from vessel_utils import (
    DEFAULT_OPENVEIN_VESSEL_CLEANUP,
    OpenVeinVesselCleanupConfig,
    vessel_cleanup_from_preset,
)

FINGER_VEIN_DATASETS: tuple[str, ...] = ("PLUS", "IDIAP", "SCUT")

# ISO capture site for finger second phalanx (Tables 1–3).
DEFAULT_CAPTURE_SITE = CaptureSite.FINGER_SECOND_PHALANX

# Q8/Q9 OpenVein vein-map cleanup preset (see README — ISO vs heuristic).
DEFAULT_VESSEL_CLEANUP_PRESET = "heuristic_default"

__all__ = [
    "FINGER_VEIN_DATASETS",
    "DEFAULT_CAPTURE_SITE",
    "DEFAULT_OPENVEIN_VESSEL_CLEANUP",
    "DEFAULT_VESSEL_CLEANUP_PRESET",
    "OpenVeinVesselCleanupConfig",
    "vessel_cleanup_from_preset",
]
