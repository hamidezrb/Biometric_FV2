"""
Run ISO Q1–Q9 + unified quality on vascular images.

Default: evaluate all OpenVein extractors (RLT, MC, WLD, PC, GF, EMC) and write:

  results/{EXTRACTOR}/csv/...
  results/final_summary/...

Examples:
  python run_all_q1_q9.py --dry-run
  python run_all_q1_q9.py
  python run_all_q1_q9.py --extractor RLT

For the production PC-only experiment (Excel/CSV), use run_finger_vein_experiment.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from vascular_quality.common.paths import (
    DEFAULT_OPENVEIN_EXTRACTOR,
    OPENVEIN_EXTRACTORS,
    PROJECT_ROOT,
)
from vascular_quality.evaluation.readiness import run_readiness_check
from vascular_quality.evaluation.runner import run_multi_extractor_evaluation
from vascular_quality.common.debug_outputs import PRODUCTION_DEBUG_EXTRACTOR
from vascular_quality.finger_vein.config import FINGER_VEIN_DATASETS
from unified_quality import unified_weight_description

RESULTS_ROOT = PROJECT_ROOT / "results"


def _resolve_datasets(dataset_arg: str) -> list[str]:
    if dataset_arg == "all":
        return list(FINGER_VEIN_DATASETS)
    if dataset_arg not in FINGER_VEIN_DATASETS:
        raise ValueError(
            f"Unknown dataset {dataset_arg!r}. "
            f"Expected one of {FINGER_VEIN_DATASETS} or 'all'."
        )
    return [dataset_arg]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="ISO/IEC 29794-9 Q1–Q9 multi-extractor evaluation pipeline",
    )
    parser.add_argument(
        "--dataset",
        default="all",
        choices=[*FINGER_VEIN_DATASETS, "all"],
        help="Finger-vein dataset(s) to evaluate (default: all).",
    )
    parser.add_argument(
        "--quality",
        default="all",
        choices=["high_quality", "low_quality", "all"],
        help="Quality class (default: all).",
    )
    parser.add_argument(
        "--extractor",
        default=None,
        help=f"Single extractor (default: all — {', '.join(OPENVEIN_EXTRACTORS)}).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_ROOT,
        help="Results root (default: results/).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan datasets and OpenVein outputs only; do not compute metrics.",
    )
    parser.add_argument(
        "--save-debug-images",
        action="store_true",
        help=(
            "Write per-image Q1–Q9 debug PNGs under "
            "debug_outputs/finger_vein/{DATASET}/{quality}/{image_stem}/ "
            f"(uses {PRODUCTION_DEBUG_EXTRACTOR} vein maps only)."
        ),
    )
    args = parser.parse_args()

    datasets = _resolve_datasets(args.dataset)
    extractors = (
        [args.extractor.upper()]
        if args.extractor
        else list(OPENVEIN_EXTRACTORS)
    )

    if args.dry_run:
        report = run_readiness_check(
            datasets=datasets,
            quality=args.quality,
            extractors=extractors,
        )
        return 0 if report.ok else 1

    print(f"Unified score: {unified_weight_description()}\n")

    _, summary = run_multi_extractor_evaluation(
        datasets=datasets,
        quality=args.quality,
        extractors=extractors,
        results_root=args.results_dir,
        save_debug_images=args.save_debug_images,
    )

    print("\n" + "=" * 60)
    print("EXECUTION STATUS:", "[OK] Success" if summary.success else "[FAIL] Failed")
    print("Total result rows:", summary.total_rows)
    if summary.warnings:
        print("\nWarnings:")
        for w in summary.warnings:
            print(f"  - {w}")
    if summary.errors:
        print("\nErrors:")
        for e in summary.errors:
            print(f"  - {e}")
    print("\nFiles generated:")
    for f in summary.files:
        if f.exists():
            print(f"  - {f.relative_to(PROJECT_ROOT)}")

    return 0 if summary.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
