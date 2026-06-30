"""Image discovery helpers."""

from __future__ import annotations

import glob
from pathlib import Path

from vascular_quality.common.paths import IMAGE_EXTENSIONS


def list_images_in_dir(directory: Path | str) -> list[Path]:
    """List image files in a directory (non-recursive), sorted by name."""
    folder = Path(directory)
    if not folder.is_dir():
        return []

    paths: list[Path] = []
    for ext in IMAGE_EXTENSIONS:
        paths.extend(Path(p) for p in glob.glob(str(folder / f"*{ext}")))
        paths.extend(Path(p) for p in glob.glob(str(folder / f"*{ext.upper()}")))
    # Deduplicate (Windows may match same file twice).
    return sorted({p.resolve() for p in paths}, key=lambda p: p.name.lower())
