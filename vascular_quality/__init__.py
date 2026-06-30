"""
Vascular image quality evaluation — modular layout by anatomy and dataset.

Subpackages:
  common       — shared paths, I/O, validation, pipeline
  openvein     — OpenVein-style preprocessing and feature extraction (Python)
  finger_vein  — PLUS / IDIAP / SCUT finger-vein experiments
  palm         — palm capture (future)
  dorsal_hand  — dorsal hand capture (future)
"""

from vascular_quality.common.paths import PROJECT_ROOT

__all__ = ["PROJECT_ROOT"]
