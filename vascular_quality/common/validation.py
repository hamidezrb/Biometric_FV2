"""Folder and file validation with clear error messages."""

from __future__ import annotations

from pathlib import Path

from vascular_quality.common.images import list_images_in_dir
from vascular_quality.common.openvein import vein_map_path
from vascular_quality.common.paths import (
    OPENVEIN_EXTRACTORS,
    QUALITY_CLASSES,
    finger_vein_dataset_names,
    finger_vein_image_dir,
    finger_vein_root,
    openvein_quality_dir,
    openvein_vein_map_dir,
)


class DatasetLayoutError(FileNotFoundError):
    """Raised when expected dataset directories are missing."""


def validate_dataset_layout(
    dataset: str,
    quality: str = "all",
    *,
    anatomy: str = "finger_vein",
) -> list[Path]:
    """
    Ensure finger-vein dataset folders exist and contain at least one image.

    Returns the list of quality class names that were validated.
    """
    if anatomy != "finger_vein":
        raise NotImplementedError(
            f"Layout validation for anatomy {anatomy!r} is not implemented yet."
        )

    datasets = finger_vein_dataset_names()
    if dataset not in datasets:
        raise DatasetLayoutError(
            f"Unknown dataset {dataset!r}. Expected one of: {', '.join(datasets)}."
        )

    root = finger_vein_root()
    if not root.is_dir():
        raise DatasetLayoutError(
            f"Finger-vein data root missing: {root}\n"
            f"Create: data/finger_vein/{{PLUS,IDIAP,SCUT}}/{{high_quality,low_quality}}/"
        )

    qualities = list(QUALITY_CLASSES) if quality == "all" else [quality]
    if quality != "all" and quality not in QUALITY_CLASSES:
        raise DatasetLayoutError(
            f"Unknown quality {quality!r}. Expected: {', '.join(QUALITY_CLASSES)} or 'all'."
        )

    missing_dirs: list[str] = []
    empty_dirs: list[str] = []

    for q in qualities:
        image_dir = finger_vein_image_dir(dataset, q)
        if not image_dir.is_dir():
            missing_dirs.append(str(image_dir))
            continue
        if not list_images_in_dir(image_dir):
            empty_dirs.append(str(image_dir))

    if missing_dirs:
        raise DatasetLayoutError(
            "Missing dataset folder(s):\n  "
            + "\n  ".join(missing_dirs)
            + "\n\nCreate the folder and add vascular images before running experiments."
        )

    if empty_dirs:
        raise DatasetLayoutError(
            "Dataset folder(s) exist but contain no images:\n  "
            + "\n  ".join(empty_dirs)
            + f"\n\nSupported extensions: .png, .jpg, .jpeg, .bmp, .tif, .tiff"
        )

    return qualities


def validate_images_present(image_dir: Path) -> list[Path]:
    images = list_images_in_dir(image_dir)
    if not images:
        raise FileNotFoundError(
            f"No images in {image_dir}. "
            f"Add PNG/JPEG/BMP/TIFF files to this folder."
        )
    return images


def validate_openvein_layout(dataset: str, quality: str = "all") -> list[str]:
    """
    Ensure debug_openvein_features/{dataset}/{quality}/{extractor}/ folders exist.

    Returns validated quality class names. Does not require PNG files inside.
    """
    from vascular_quality.common.paths import OPENVEIN_DATASETS, iter_quality_classes

    if dataset not in OPENVEIN_DATASETS:
        raise DatasetLayoutError(
            f"Unknown dataset {dataset!r} for OpenVein layout. "
            f"Expected: {', '.join(OPENVEIN_DATASETS)}."
        )

    qualities = list(iter_quality_classes(quality))
    missing: list[str] = []

    for q in qualities:
        quality_dir = openvein_quality_dir(dataset, q)
        if not quality_dir.is_dir():
            missing.append(str(quality_dir))
            continue
        for extractor in OPENVEIN_EXTRACTORS:
            extractor_dir = openvein_vein_map_dir(dataset, q, extractor)
            if not extractor_dir.is_dir():
                missing.append(str(extractor_dir))

    if missing:
        raise DatasetLayoutError(
            "Missing OpenVein folder(s):\n  "
            + "\n  ".join(missing)
            + "\n\nRun: python scripts/setup_finger_vein_layout.py"
        )

    return qualities


def validate_vein_maps_present(
    dataset: str,
    quality: str,
    extractor: str,
    image_paths: list[Path],
) -> Path:
    """Ensure vein-map directory exists and every input image has a matching map."""
    if extractor not in OPENVEIN_EXTRACTORS:
        raise ValueError(
            f"Unknown extractor {extractor!r}. "
            f"Expected one of: {', '.join(OPENVEIN_EXTRACTORS)}."
        )

    vein_dir = openvein_vein_map_dir(dataset, quality, extractor)
    if not vein_dir.is_dir():
        raise FileNotFoundError(
            f"Vein-map folder missing: {vein_dir}\n"
            f"Run: python -m vascular_quality.openvein.pipeline "
            f"--backend matlab --dataset {dataset} --quality {quality} "
            f"--extractors {extractor}\n"
            f"Or create:\n"
            f"  debug_openvein_features/{dataset}/{quality}/{extractor}/\n"
            f"with one vein map per input image (same filename)."
        )

    missing: list[str] = []
    for image_path in image_paths:
        vein_path = vein_map_path(vein_dir, image_path)
        if not vein_path.is_file():
            missing.append(image_path.name)

    if missing:
        preview = ", ".join(missing[:5])
        suffix = f" ... (+{len(missing) - 5} more)" if len(missing) > 5 else ""
        raise FileNotFoundError(
            f"Missing {len(missing)} vein map(s) in {vein_dir}:\n"
            f"  {preview}{suffix}\n"
            f"Each input image needs a matching file in the extractor folder."
        )

    return vein_dir
