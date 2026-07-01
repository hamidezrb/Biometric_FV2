"""Per-image Q1–Q9 debug output layout and validation."""

from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np

from vascular_quality.common.images import list_images_in_dir
from vascular_quality.common.paths import (
    DEBUG_OUTPUTS_DIR,
    OPENVEIN_DATASETS,
    finger_vein_image_dir,
    iter_quality_classes,
)

from vascular_quality.common.visualization import overlay_mask, visualize_feature_points_iso_style
from q6 import Q6Result, build_q6_debug_images

# Same datasets as finger_vein.config; avoid importing that package here (circular import).
FINGER_VEIN_DATASETS = OPENVEIN_DATASETS

# Exactly one PNG per Q metric inside each image folder.
Q_DEBUG_FILENAMES: tuple[str, ...] = tuple(f"q{i}_debug.png" for i in range(1, 10))

PRODUCTION_DEBUG_EXTRACTOR = "PC"

PC_VESSEL_DEBUG_FILENAMES: tuple[str, ...] = (
    "original.png",
    "pc_feature_raw.png",
    "pc_binary_before_cleaning.png",
    "pc_binary_after_cleaning.png",
    "pc_skeleton.png",
    "q8_skeleton_used.png",
    "q9_feature_points.png",
)

Q6_DEBUG_FILENAMES: tuple[str, ...] = (
    "q6_original.png",
    "q6_foreground_mask.png",
    "q6_sobel_response.png",
    "q6_edge_pixels.png",
)


def finger_vein_image_debug_dir(dataset: str, quality: str, image_stem: str) -> Path:
    """debug_outputs/finger_vein/{dataset}/{quality}/{image_stem}/"""
    return finger_vein_debug_quality_dir(dataset, quality) / image_stem


def finger_vein_debug_quality_dir(dataset: str, quality: str) -> Path:
    """debug_outputs/finger_vein/{dataset}/{quality}/"""
    return DEBUG_OUTPUTS_DIR / "finger_vein" / dataset / quality


def write_q_debug_images(
    out_dir: Path,
    *,
    gray: np.ndarray,
    r_mask: np.ndarray,
    unoccluded_mask: np.ndarray,
    vein_img: np.ndarray,
    skel_q8: np.ndarray,
    skel_q9: np.ndarray,
    q9_points_vis: np.ndarray,
    q6_result: Q6Result | None = None,
) -> list[Path]:
    """Write q1_debug.png … q9_debug.png into ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    q1_vis = overlay_mask(gray, r_mask)
    q2_vis = overlay_mask(gray, unoccluded_mask, color=(255, 128, 0))
    q3_vis = cv2.applyColorMap(
        cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
        cv2.COLORMAP_VIRIDIS,
    )
    masked = gray.copy()
    masked[r_mask != 255] = 0
    q4_vis = cv2.cvtColor(masked, cv2.COLOR_GRAY2BGR)
    q5_vis = overlay_mask(gray, r_mask, color=(0, 200, 255))
    if q6_result is None:
        from q6 import calculate_q6_detailed

        q6_result = calculate_q6_detailed(
            r_mask,
            gray,
            S_unoccluded=int(np.count_nonzero(r_mask == 255)),
        )
    q6_debug = build_q6_debug_images(gray, r_mask, q6_result)
    q6_vis = q6_debug["q6_edge_pixels.png"]
    q7_vis = cv2.applyColorMap(
        cv2.normalize(masked, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
        cv2.COLORMAP_PLASMA,
    )
    q8_vis = cv2.cvtColor((skel_q8.astype(np.uint8) * 255), cv2.COLOR_GRAY2BGR)
    q9_vis = q9_points_vis

    images = [q1_vis, q2_vis, q3_vis, q4_vis, q5_vis, q6_vis, q7_vis, q8_vis, q9_vis]
    for name, img in zip(Q_DEBUG_FILENAMES, images):
        path = out_dir / name
        cv2.imwrite(str(path), img)
        written.append(path)
    for name, img in q6_debug.items():
        path = out_dir / name
        cv2.imwrite(str(path), img)
        written.append(path)
    return written


def write_pc_vessel_debug_stages(
    out_dir: Path,
    *,
    original: np.ndarray,
    stages: dict[str, np.ndarray],
    skel_q8: np.ndarray,
    q9_points_vis: np.ndarray,
) -> list[Path]:
    """Write OpenVein PC vessel-processing debug stages for Q8/Q9 inspection."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    mapping = {
        "original.png": original,
        "pc_feature_raw.png": stages["pc_feature_raw"],
        "pc_binary_before_cleaning.png": stages["pc_binary_before_cleaning"],
        "pc_binary_after_cleaning.png": stages["pc_binary_after_cleaning"],
        "pc_skeleton.png": stages["pc_skeleton"],
        "q8_skeleton_used.png": (skel_q8.astype(np.uint8) * 255),
        "q9_feature_points.png": q9_points_vis,
    }
    for name, img in mapping.items():
        path = out_dir / name
        if img.ndim == 2:
            cv2.imwrite(str(path), img)
        else:
            cv2.imwrite(str(path), img)
        written.append(path)
    return written


def remove_stale_debug_folders(
    datasets: list[str] | None = None,
    quality: str = "all",
) -> list[str]:
    """
    Remove debug folders under debug_outputs/finger_vein/ that do not match
    a current input image stem. Also remove legacy flat PNG files in quality dirs.
    """
    datasets = datasets or list(FINGER_VEIN_DATASETS)
    removed: list[str] = []

    for dataset in datasets:
        for q in iter_quality_classes(quality):
            quality_dir = finger_vein_debug_quality_dir(dataset, q)
            if not quality_dir.is_dir():
                continue

            expected = {p.stem for p in list_images_in_dir(finger_vein_image_dir(dataset, q))}

            # Legacy flat files: {stem}_gray.png etc.
            for png in quality_dir.glob("*.png"):
                removed.append(str(png))
                png.unlink()

            for child in quality_dir.iterdir():
                if child.is_file():
                    continue
                if not child.is_dir():
                    continue
                if child.name not in expected:
                    shutil.rmtree(child)
                    removed.append(str(child))

    # Remove obsolete _runs/ and extractor-grouped legacy trees if empty junk
    legacy_runs = DEBUG_OUTPUTS_DIR / "finger_vein" / "_runs"
    if legacy_runs.is_dir():
        shutil.rmtree(legacy_runs)
        removed.append(str(legacy_runs))

    return removed


def validate_debug_layout(
    datasets: list[str] | None = None,
    quality: str = "all",
) -> dict[str, list[str]]:
    """Return issues keyed by dataset/quality for debug folder layout."""
    datasets = datasets or list(FINGER_VEIN_DATASETS)
    issues: dict[str, list[str]] = {}

    for dataset in datasets:
        for q in iter_quality_classes(quality):
            key = f"{dataset}/{q}"
            image_dir = finger_vein_image_dir(dataset, q)
            quality_debug = finger_vein_debug_quality_dir(dataset, q)
            images = list_images_in_dir(image_dir) if image_dir.is_dir() else []
            stems = {p.stem for p in images}
            problems: list[str] = []

            if not quality_debug.is_dir() and images:
                problems.append("missing quality debug directory")
                issues[key] = problems
                continue

            debug_dirs = (
                [d for d in quality_debug.iterdir() if d.is_dir()]
                if quality_debug.is_dir()
                else []
            )
            debug_stems = {d.name for d in debug_dirs}

            missing = sorted(stems - debug_stems)
            extra = sorted(debug_stems - stems)
            if missing:
                problems.append(f"missing folders: {', '.join(missing)}")
            if extra:
                problems.append(f"extra folders: {', '.join(extra)}")

            for stem in stems:
                folder = quality_debug / stem
                if not folder.is_dir():
                    continue
                for qfile in Q_DEBUG_FILENAMES:
                    if not (folder / qfile).is_file():
                        problems.append(f"{stem}: missing {qfile}")
                extras = [
                    f.name
                    for f in folder.iterdir()
                    if f.name not in Q_DEBUG_FILENAMES
                    and f.name not in PC_VESSEL_DEBUG_FILENAMES
                    and f.name not in Q6_DEBUG_FILENAMES
                ]
                if extras:
                    problems.append(f"{stem}: unexpected files: {', '.join(extras)}")

            if problems:
                issues[key] = problems

    return issues
