"""
Create finger-vein dataset and OpenVein debug folder layout.

Run once after cloning:
  python scripts/setup_finger_vein_layout.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow imports when executed as scripts/setup_finger_vein_layout.py
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from vascular_quality.common.paths import (
    DEBUG_OPENVEIN_DIR,
    DEBUG_OUTPUTS_DIR,
    OPENVEIN_EXTRACTORS,
    PROJECT_ROOT,
    QUALITY_CLASSES,
    ensure_dir,
    finger_vein_root,
    openvein_quality_dir,
    openvein_vein_map_dir,
)
from vascular_quality.finger_vein.config import FINGER_VEIN_DATASETS


def create_layout() -> list[Path]:
    created: list[Path] = []

    for dataset in FINGER_VEIN_DATASETS:
        for quality in QUALITY_CLASSES:
            image_dir = ensure_dir(finger_vein_root() / dataset / quality)
            created.append(image_dir)

            for extractor in OPENVEIN_EXTRACTORS:
                vein_dir = ensure_dir(openvein_vein_map_dir(dataset, quality, extractor))
                created.append(vein_dir)

    # Optional debug visualization root (metric PNGs written only with --save-debug-images).
    created.append(ensure_dir(DEBUG_OUTPUTS_DIR / "finger_vein" / "_runs"))

    # Reserved anatomy roots (future).
    for anatomy in ("palm", "dorsal_hand"):
        for quality in QUALITY_CLASSES:
            created.append(ensure_dir(PROJECT_ROOT / "data" / anatomy / quality))

    return created


def migrate_legacy_test_images() -> list[tuple[str, str]]:
    """Copy test_images/* into data/finger_vein/PLUS/ if legacy folder exists."""
    moved: list[tuple[str, str]] = []
    legacy = PROJECT_ROOT / "test_images"
    if not legacy.is_dir():
        return moved

    for quality in QUALITY_CLASSES:
        src = legacy / quality
        if not src.is_dir():
            continue
        dst = finger_vein_root() / "PLUS" / quality
        ensure_dir(dst)
        for src_file in src.iterdir():
            if not src_file.is_file():
                continue
            dst_file = dst / src_file.name
            if not dst_file.exists():
                import shutil

                shutil.copy2(src_file, dst_file)
                moved.append((str(src_file), str(dst_file)))
    return moved


def migrate_legacy_openvein_flat() -> list[tuple[str, str]]:
    """
    Copy old flat debug_openvein_features/{RLT,MC,...} into
    debug_openvein_features/PLUS/{quality}/{extractor}/ when possible.
    """
    moved: list[tuple[str, str]] = []
    if not DEBUG_OPENVEIN_DIR.is_dir():
        return moved

    for extractor in OPENVEIN_EXTRACTORS:
        flat = DEBUG_OPENVEIN_DIR / extractor
        if not flat.is_dir():
            continue
        for png in flat.glob("*.png"):
            # Heuristic: filenames containing PALMAR were in low/high test sets.
            quality = "low_quality" if "018" in png.name or "026" in png.name else "high_quality"
            dst_dir = ensure_dir(openvein_vein_map_dir("PLUS", quality, extractor))
            dst = dst_dir / png.name
            if not dst.exists():
                import shutil

                shutil.copy2(png, dst)
                moved.append((str(png), str(dst)))
    return moved


def cleanup_legacy_flat_openvein_dirs() -> list[str]:
    """
    Remove obsolete flat extractor folders at debug_openvein_features/{RLT,MC,...}/.

    Only removes a folder when it is empty or after files were migrated to
    debug_openvein_features/{DATASET}/{quality}/{extractor}/.
    """
    import shutil

    removed: list[str] = []
    if not DEBUG_OPENVEIN_DIR.is_dir():
        return removed

    for extractor in OPENVEIN_EXTRACTORS:
        flat = DEBUG_OPENVEIN_DIR / extractor
        if not flat.is_dir():
            continue
        # Skip if any PNGs remain (not yet migrated).
        if any(flat.glob("*.png")):
            continue
        shutil.rmtree(flat)
        removed.append(str(flat))
    return removed


def print_summary(created: list[Path], migrated_images, migrated_veins, removed_flat) -> None:
    print("=" * 60)
    print("Finger-vein project layout ready")
    print("=" * 60)
    print(f"\nProject root: {PROJECT_ROOT}")
    print("\nExpected image layout:")
    print("  data/finger_vein/{PLUS,IDIAP,SCUT}/{high_quality,low_quality}/*.png")
    print("\nExpected OpenVein vein maps:")
    print("  debug_openvein_features/{DATASET}/{quality}/{RLT,MC,WLD,PC,GF,EMC}/")
    print("\nISO debug visualizations (optional, off by default):")
    print("  debug_outputs/finger_vein/_runs/{timestamp}/...")
    print(f"\nDirectories ensured: {len(created)}")

    if migrated_images:
        print(f"\nMigrated {len(migrated_images)} image(s) from test_images/ -> data/finger_vein/PLUS/")
    if migrated_veins:
        print(f"Migrated {len(migrated_veins)} vein map(s) to debug_openvein_features/PLUS/...")
    if removed_flat:
        print(f"Removed {len(removed_flat)} legacy flat extractor folder(s).")

    print("\nPC quality experiment (after PC vein maps exist):")
    print("  python run_finger_vein_experiment.py --extractor PC --datasets PLUS IDIAP SCUT --qualities high_quality low_quality --output results/finger_vein/PC --save-excel --dry-run")
    print("\nGenerate vein maps (Python + MATLAB Engine / OpenVein):")
    print("  python -m vascular_quality.openvein.pipeline --backend matlab --dataset PLUS --quality high_quality --extractors PC --matlab-toolkit-root <OpenVein-Toolkit>")
    print("=" * 60)


def main() -> int:
    migrated_images = migrate_legacy_test_images()
    migrated_veins = migrate_legacy_openvein_flat()
    removed_flat = cleanup_legacy_flat_openvein_dirs()
    created = create_layout()
    print_summary(created, migrated_images, migrated_veins, removed_flat)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
