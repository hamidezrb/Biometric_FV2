"""OpenVein feature-map path helpers."""

from __future__ import annotations

from pathlib import Path

from vascular_quality.common.paths import (
    DEFAULT_OPENVEIN_EXTRACTOR,
    OPENVEIN_EXTRACTORS,
    ensure_dir,
    openvein_vein_map_dir,
)


def vein_map_path(
    vein_root: Path | str,
    image_path: Path | str,
) -> Path:
    """
    Resolve vein-map file for an input image.

    OpenVein always writes feature maps as ``{stem}.png`` regardless of input
    extension (.bmp, .jpg, etc.).
    """
    src = Path(image_path)
    return Path(vein_root) / f"{src.stem}.png"


def vein_map_basename(image_path: Path | str) -> str:
    """Feature map filename for an input vascular image."""
    return f"{Path(image_path).stem}.png"


def openvein_output_dir(
    dataset: str,
    quality: str,
    extractor: str = DEFAULT_OPENVEIN_EXTRACTOR,
) -> Path:
    """Target directory for one OpenVein extractor run."""
    return ensure_dir(openvein_vein_map_dir(dataset, quality, extractor))


def list_extractor_dirs(dataset: str, quality: str) -> list[Path]:
    """Return existing extractor subfolders for a dataset/quality pair."""
    from vascular_quality.common.paths import openvein_quality_dir

    base = openvein_quality_dir(dataset, quality)
    if not base.is_dir():
        return []
    return sorted(
        p for p in base.iterdir()
        if p.is_dir() and p.name in OPENVEIN_EXTRACTORS
    )
