"""
Run ISO Q1–Q9 + unified quality on finger-vein datasets (PLUS / IDIAP / SCUT).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from tabulate import tabulate

from unified_quality import DEFAULT_POWER_COEFFICIENTS, format_power_coefficients
from vascular_quality.common.images import list_images_in_dir
from vascular_quality.common.debug_outputs import (
    finger_vein_image_debug_dir,
    remove_stale_debug_folders,
)
from vascular_quality.common.paths import (
    DEFAULT_OPENVEIN_EXTRACTOR,
    ensure_dir,
    finger_vein_image_dir,
    iter_quality_classes,
    openvein_vein_map_dir,
)
from vascular_quality.common.pipeline import run_q1_q9_on_image
from vascular_quality.common.validation import (
    validate_dataset_layout,
    validate_openvein_layout,
    validate_vein_maps_present,
)
from vascular_quality.finger_vein.config import (
    DEFAULT_CAPTURE_SITE,
    DEFAULT_VESSEL_CLEANUP_PRESET,
    FINGER_VEIN_DATASETS,
    OpenVeinVesselCleanupConfig,
    vessel_cleanup_from_preset,
)


def collect_finger_vein_images(
    dataset: str,
    quality: str,
) -> list[tuple[str, Path]]:
    """
    Return (quality_class, image_path) pairs for a dataset run.

    quality may be 'high_quality', 'low_quality', or 'all'.
    """
    pairs: list[tuple[str, Path]] = []
    for q in iter_quality_classes(quality):
        image_dir = finger_vein_image_dir(dataset, q)
        for path in list_images_in_dir(image_dir):
            pairs.append((q, path))
    return pairs


def run_finger_vein_batch(
    dataset: str,
    quality: str = "all",
    *,
    extractor: str = DEFAULT_OPENVEIN_EXTRACTOR,
    out_root: Path | None = None,
    skip_validation: bool = False,
    save_debug_images: bool = False,
    openvein_cleanup: OpenVeinVesselCleanupConfig | None = None,
) -> list[dict[str, Any]]:
    """
    Process all images for one finger-vein dataset and quality class(es).

    Raises FileNotFoundError / DatasetLayoutError when folders or vein maps
    are missing (unless skip_validation=True).
    """
    if dataset not in FINGER_VEIN_DATASETS:
        raise ValueError(
            f"Unknown dataset {dataset!r}. Choose from: {', '.join(FINGER_VEIN_DATASETS)}."
        )

    if not skip_validation:
        validate_dataset_layout(dataset, quality, anatomy="finger_vein")
        validate_openvein_layout(dataset, quality)

    cleanup = openvein_cleanup or vessel_cleanup_from_preset(DEFAULT_VESSEL_CLEANUP_PRESET)

    if save_debug_images:
        remove_stale_debug_folders([dataset], quality)

    image_pairs = collect_finger_vein_images(dataset, quality)
    if not image_pairs:
        raise FileNotFoundError(
            f"No images found for dataset={dataset}, quality={quality}."
        )

    # Group by quality so vein-map folders are checked per class.
    results: list[dict[str, Any]] = []
    by_quality: dict[str, list[Path]] = {}
    for q, path in image_pairs:
        by_quality.setdefault(q, []).append(path)

    for q, paths in by_quality.items():
        vein_root = openvein_vein_map_dir(dataset, q, extractor)
        if not skip_validation:
            validate_vein_maps_present(dataset, q, extractor, paths)

        for image_path in paths:
            out_dir = (
                finger_vein_image_debug_dir(dataset, q, image_path.stem)
                if save_debug_images
                else None
            )
            try:
                row = run_q1_q9_on_image(
                    image_path,
                    vein_root,
                    out_dir,
                    capture_site=DEFAULT_CAPTURE_SITE,
                    save_debug_images=save_debug_images,
                    openvein_cleanup=cleanup,
                )
                row["dataset"] = dataset
                row["quality"] = q
                row["extractor"] = extractor
                results.append(row)
            except Exception as exc:
                print(f"ERROR on {image_path}: {exc}")

    return results


def _print_results_table(results: list[dict[str, Any]]) -> None:
    headers = [
        "dataset",
        "quality",
        "image",
        "Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8", "Q9",
        "unified_quality_score",
        "N_vessel", "N_end", "N_int", "N_fp",
    ]
    table = []
    for r in results:
        table.append([
            r.get("dataset", ""),
            r.get("quality", ""),
            r["file"],
            r["Q1"], r["Q2"], r["Q3"], r["Q4"], r["Q5"],
            r["Q6"], r["Q7"], r["Q8"], r["Q9"],
            r["unified_quality_score"],
            r["N_vessel"], r["N_end"], r["N_int"], r["N_fp"],
        ])

    coeffs = (
        results[0]["power_coefficients"]
        if results
        else DEFAULT_POWER_COEFFICIENTS
    )

    print("\nFINGER VEIN QUALITY SCORES (ISO/IEC 29794-9)\n")
    print("Power coefficients α_i (ISO 5.3):")
    print(f"  {format_power_coefficients(coeffs)}")
    print("  (placeholder 1/9 each — import calibrated values when available)\n")

    if table:
        print(tabulate(table, headers=headers, tablefmt="github"))
    else:
        print("No images processed successfully.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Finger-vein ISO Q1–Q9 quality experiment (PLUS / IDIAP / SCUT).",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=list(FINGER_VEIN_DATASETS),
        help="Finger-vein dataset name.",
    )
    parser.add_argument(
        "--quality",
        default="all",
        choices=["high_quality", "low_quality", "all"],
        help="Quality class folder to process (default: all).",
    )
    parser.add_argument(
        "--extractor",
        default=DEFAULT_OPENVEIN_EXTRACTOR,
        help=f"OpenVein submodel folder (default: {DEFAULT_OPENVEIN_EXTRACTOR}).",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Optional single image path (overrides dataset folder scan).",
    )
    parser.add_argument(
        "--vein-root",
        default=None,
        help="Optional vein-map folder (overrides debug_openvein_features layout).",
    )
    parser.add_argument(
        "--save-debug-images",
        action="store_true",
        help=(
            "Write per-image Q1–Q9 debug PNGs under "
            "debug_outputs/finger_vein/{DATASET}/{quality}/{image_stem}/."
        ),
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Deprecated: use --save-debug-images (writes to timestamped _runs/ folder).",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Do not check folder layout before processing.",
    )
    parser.add_argument(
        "--vessel-cleanup",
        default=DEFAULT_VESSEL_CLEANUP_PRESET,
        choices=["iso_minimal", "heuristic_default"],
        help=(
            "Q8/Q9 OpenVein vein-map preprocessing preset "
            "(iso_minimal = ISO steps b–c only; heuristic_default = default)."
        ),
    )
    args = parser.parse_args(argv)

    cleanup = vessel_cleanup_from_preset(args.vessel_cleanup)

    save_debug = args.save_debug_images or args.out is not None

    if args.input:
        image_path = Path(args.input)
        if not image_path.is_file():
            parser.error(f"Input image not found: {image_path}")

        quality = args.quality if args.quality != "all" else "high_quality"
        vein_root = Path(args.vein_root) if args.vein_root else openvein_vein_map_dir(
            args.dataset, quality, args.extractor
        )
        out_dir = (
            Path(args.out)
            if args.out
            else finger_vein_image_debug_dir(args.dataset, quality, image_path.stem)
            if save_debug
            else None
        )

        try:
            results = [
                run_q1_q9_on_image(
                    image_path,
                    vein_root,
                    out_dir,
                    capture_site=DEFAULT_CAPTURE_SITE,
                    save_debug_images=save_debug,
                    openvein_cleanup=cleanup,
                )
            ]
            results[0]["dataset"] = args.dataset
            results[0]["quality"] = quality
            results[0]["extractor"] = args.extractor
        except Exception as exc:
            print(f"ERROR: {exc}")
            return 1
    else:
        try:
            results = run_finger_vein_batch(
                args.dataset,
                args.quality,
                extractor=args.extractor,
                skip_validation=args.skip_validation,
                save_debug_images=save_debug,
                openvein_cleanup=cleanup,
            )
        except (FileNotFoundError, ValueError) as exc:
            print(f"ERROR: {exc}")
            return 1

    _print_results_table(results)
    return 0 if results else 1


if __name__ == "__main__":
    raise SystemExit(main())
