"""Full-hand vascular capture — ISO Tables 1–3 FULL_HAND coefficients."""

from __future__ import annotations

from iso_constants import CaptureSite

# Full hand: Sc=300000, Lc=9000, Fc=50 (iso_constants.CAPTURE_SITE_COEFFICIENTS).
DEFAULT_CAPTURE_SITE = CaptureSite.FULL_HAND

__all__ = ["DEFAULT_CAPTURE_SITE"]
