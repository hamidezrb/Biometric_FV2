"""
Vascular image quality evaluation — modular layout by anatomy and dataset.

Subpackages:
  common       — shared paths, I/O, validation, pipeline
  openvein     — OpenVein-style preprocessing and feature extraction (Python)
  finger_vein  — PLUS / IDIAP / SCUT finger-vein experiments
  palm         — palm / dorsal ROI capture (PALM_OR_DORSAL coefficients)
  dorsal_hand  — dorsal hand ROI capture (PALM_OR_DORSAL coefficients)
  full_hand    — full-hand capture (FULL_HAND coefficients)
"""

from vascular_quality.common.paths import PROJECT_ROOT

__all__ = ["PROJECT_ROOT"]
