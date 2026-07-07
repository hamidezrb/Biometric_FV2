"""Image I/O for OpenVein-style feature extraction."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from vascular_quality.common.images import list_images_in_dir
from vascular_quality.common.openvein import vein_map_basename
from vascular_quality.common.paths import ensure_dir, openvein_vein_map_dir


def load_grayscale(image_path: Path | str) -> np.ndarray:
    """Load a single-channel grayscale image."""
    path = Path(image_path)
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"Could not read image: {path}")
    return gray


def save_feature_image(
    feature: np.ndarray,
    output_path: Path | str,
) -> Path:
    """
    Save a binary vein feature map as uint8 PNG (0/255).

    Matches OpenVein MATLAB export: ``uint8(feat) * 255``.
    """
    out = Path(output_path)
    ensure_dir(out.parent)

    if feature.dtype == bool:
        feat8 = feature.astype(np.uint8) * 255
    elif feature.dtype in (np.float32, np.float64):
        feat8 = (np.clip(feature, 0.0, 1.0) * 255).astype(np.uint8)
    else:
        feat8 = np.where(feature > 0, 255, 0).astype(np.uint8)

    if not cv2.imwrite(str(out), feat8):
        raise OSError(f"Failed to write feature image: {out}")
    return out


def list_input_images(input_dir: Path | str) -> list[Path]:
    """List images in stable sorted order (same basename preserved on save)."""
    folder = Path(input_dir)
    if not folder.is_dir():
        raise FileNotFoundError(
            f"Input folder does not exist: {folder}\n"
            f"Expected layout: data/{{modality}}/{{DATASET}}/{{high_quality|low_quality}}/"
        )
    images = list_images_in_dir(folder)
    if not images:
        raise FileNotFoundError(
            f"No images in {folder}. "
            f"Add PNG/JPEG/BMP/TIFF files before running extraction."
        )
    return images


def extractor_output_dir(
    output_root: Path | str,
    modality: str,
    dataset: str,
    quality: str,
    extractor: str,
) -> Path:
    """Resolve and create the OpenVein extractor output directory for one run."""
    return ensure_dir(
        openvein_vein_map_dir(
            dataset,
            quality,
            extractor,
            modality=modality,
            output_root=Path(output_root),
        )
    )


def expected_output_names(input_images: list[Path]) -> set[str]:
    """Basenames of feature PNGs that should exist for the given inputs."""
    return {vein_map_basename(p) for p in input_images}


def list_output_images(output_dir: Path | str) -> list[Path]:
    """List PNG feature maps in an extractor folder (non-recursive), sorted by name."""
    folder = Path(output_dir)
    if not folder.is_dir():
        return []
    return sorted(folder.glob("*.png"), key=lambda p: p.name.lower())


def find_stale_outputs(
    output_dir: Path | str,
    input_images: list[Path],
) -> list[Path]:
    """Output PNGs whose names do not match any current input image."""
    expected = expected_output_names(input_images)
    return [p for p in list_output_images(output_dir) if p.name not in expected]


def report_stale_outputs(
    output_dir: Path,
    input_images: list[Path],
    *,
    extractor: str,
) -> list[Path]:
    """Print stale files clearly; return the stale path list (does not delete)."""
    stale = find_stale_outputs(output_dir, input_images)
    if not stale:
        return stale
    print(f"\nSTALE OUTPUT FILES ({extractor}) in {output_dir}:")
    print("  These files do not match any current input image and may be from a prior run:")
    for path in stale:
        print(f"    {path.name}")
    print("  Use --clean-output to remove stale files before extraction.")
    return stale


def remove_stale_outputs(
    output_dir: Path,
    input_images: list[Path],
    *,
    extractor: str,
) -> list[Path]:
    """Remove stale output PNGs; log each removal explicitly."""
    stale = find_stale_outputs(output_dir, input_images)
    if not stale:
        return stale
    print(f"\nRemoving {len(stale)} stale file(s) from {extractor}/ before extraction:")
    for path in stale:
        print(f"  delete: {path.name}")
        path.unlink()
    return stale


def validate_input_output_mapping(
    input_images: list[Path],
    output_dir: Path,
    *,
    extractor: str,
    strict: bool = False,
) -> bool:
    """
    Verify one-to-one mapping between inputs and output PNGs.

    Returns True when valid. Prints WARNING on mismatch (raises if strict=True).
    """
    expected = expected_output_names(input_images)
    outputs = list_output_images(output_dir)
    output_names = {p.name for p in outputs}
    stale = [p for p in outputs if p.name not in expected]
    missing = sorted(expected - output_names)

    ok = (
        len(outputs) == len(input_images)
        and not stale
        and not missing
    )

    if ok:
        return True

    print(f"\nWARNING ({extractor}):")
    print(f"  Input images:     {len(input_images)}")
    print(f"  Generated images: {len(outputs)}")
    if stale:
        print(f"  Stale outputs ({len(stale)}): {', '.join(p.name for p in stale)}")
    if missing:
        print(f"  Missing outputs ({len(missing)}): {', '.join(missing)}")
    print("\nPossible causes:")
    print("  - stale output files from a previous run (output dirs are not auto-cleaned)")
    print("  - duplicate processing")
    print("  - incorrect filename generation")
    print("  - cached / leftover debug results")
    print("  Re-run with --clean-output to remove stale files before extraction.")

    if strict:
        raise ValueError(
            f"Input/output count mismatch for {extractor}: "
            f"{len(input_images)} input(s), {len(outputs)} output(s)."
        )
    return False


def collect_written_outputs(
    output_dir: Path,
    input_images: list[Path],
) -> list[Path]:
    """Return output paths that match current input stems (ignores stale files)."""
    return [
        output_dir / vein_map_basename(img)
        for img in input_images
        if (output_dir / vein_map_basename(img)).is_file()
    ]


def _convert_image_to_png(src: Path, dst: Path) -> None:
    gray = load_grayscale(src)
    if not cv2.imwrite(str(dst), gray):
        raise OSError(f"Failed to convert {src} -> {dst}")


def prepare_matlab_input_dir(
    image_dir: Path,
    images: list[Path],
) -> tuple[Path, Path | None]:
    """
    Return (input_dir_for_matlab, cleanup_dir).

    OpenVein MATLAB reads ``*.png`` only. Non-PNG inputs (e.g. SCUT ``.bmp``) are
    converted to ``{stem}.png`` in a staging folder. Also stages when ``--limit``
    selects a subset of a folder.
    """
    folder_images = list_images_in_dir(image_dir)
    folder_set = set(folder_images)
    selected = list(images)

    needs_staging = (
        set(selected) != folder_set
        or len(selected) != len(folder_images)
        or any(p.suffix.lower() != ".png" for p in selected)
    )

    if not needs_staging:
        return image_dir, None

    import shutil
    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="openvein_matlab_in_"))
    for src in selected:
        dst = tmp / vein_map_basename(src)
        if src.suffix.lower() == ".png":
            shutil.copy2(src, dst)
        else:
            _convert_image_to_png(src, dst)
    return tmp, tmp


def prepare_limited_input_dir(
    image_dir: Path,
    images: list[Path],
) -> tuple[Path, Path | None]:
    """Backward-compatible alias for :func:`prepare_matlab_input_dir`."""
    return prepare_matlab_input_dir(image_dir, images)
