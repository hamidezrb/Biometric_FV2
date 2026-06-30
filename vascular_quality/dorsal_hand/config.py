"""Dorsal hand vascular capture — reserved for future experiments."""

from __future__ import annotations

from iso_constants import CaptureSite

# Dorsal hand uses palm/dorsal table coefficients in the PWI draft.
DEFAULT_CAPTURE_SITE = CaptureSite.PALM_OR_DORSAL

__all__ = ["DEFAULT_CAPTURE_SITE"]
