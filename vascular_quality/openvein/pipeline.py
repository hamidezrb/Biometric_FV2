"""
OpenVein feature extraction pipeline (CLI + experiment manager).

Architecture::

    Dataset -> Python pipeline -> backend (matlab|python) -> feature images
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from vascular_quality.common.paths import (
    DEBUG_OPENVEIN_DIR,
    PROJECT_ROOT,
    QUALITY_CLASSES,
    iter_quality_classes,
)
from vascular_quality.openvein.backend import (
    DEFAULT_BACKEND,
    DEFAULT_MATLAB_TOOLKIT_ENV,
    BackendKind,
    get_backend,
    resolve_extraction_job,
    resolve_matlab_toolkit_root,
)
from vascular_quality.openvein.matlab_backend import (
    MatlabEngineNotInstalled,
    MatlabExtractorError,
    MatlabToolkitNotFound,
)
from vascular_quality.openvein.extractors import (
    EXTRACTOR_NAMES,
    ExtractorUnavailable,
    list_extractors,
)


def run_extraction(
    *,
    backend: BackendKind = DEFAULT_BACKEND,
    dataset: str | None = None,
    quality: str = "high_quality",
    input_dir: Path | str | None = None,
    output_root: Path | str | None = None,
    extractors: list[str] | None = None,
    matlab_toolkit_root: Path | str | None = None,
    continue_on_error: bool = False,
    dry_run: bool = False,
    limit: int | None = None,
    skip_unavailable_preprocess: bool = True,
    skip_unavailable_extractors: bool = True,
    clean_output: bool = False,
) -> dict[str, list[Path]]:
    """
    Run OpenVein feature extraction through the selected backend.

    Returns mapping extractor_tag -> list of written output paths.
    """
    job = resolve_extraction_job(
        dataset=dataset,
        quality=quality,
        input_dir=Path(input_dir) if input_dir else None,
        output_root=output_root,
        extractors=extractors,
        limit=limit,
    )
    toolkit_root = resolve_matlab_toolkit_root(
        str(matlab_toolkit_root) if matlab_toolkit_root else None
    )
    impl = get_backend(backend)

    if dry_run:
        impl.dry_run(job, toolkit_root=toolkit_root)
        return {tag: [] for tag in job.extractors}

    return impl.run(
        job,
        toolkit_root=toolkit_root,
        continue_on_error=continue_on_error,
        skip_unavailable_preprocess=skip_unavailable_preprocess,
        skip_unavailable_extractors=skip_unavailable_extractors,
        clean_output=clean_output,
    )


def _print_extractor_status() -> None:
    print("OpenVein extractors:")
    print("  MATLAB backend (--backend matlab): all extractors use original OpenVein.")
    print("  Python backend (--backend python): approximate / placeholder only.\n")
    for spec in list_extractors():
        status = spec.parity.value
        suffix = f" — {spec.notes}" if spec.notes else ""
        print(f"  {spec.tag:3s} {spec.full_name:25s} [python: {status}]{suffix}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "OpenVein vascular feature extraction. "
            "Default backend: MATLAB Engine (original OpenVein algorithms)."
        ),
    )
    parser.add_argument(
        "--backend",
        choices=("matlab", "python"),
        default=DEFAULT_BACKEND,
        help="matlab = original OpenVein via Engine API (default); python = approximate fallback.",
    )
    parser.add_argument(
        "--matlab-toolkit-root",
        type=Path,
        default=None,
        help=(
            "Path to OpenVein-Toolkit install (required for --backend matlab). "
            f"Or set {DEFAULT_MATLAB_TOOLKIT_ENV}."
        ),
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset: PLUS, IDIAP, or SCUT (or path data/finger_vein/PLUS).",
    )
    parser.add_argument(
        "--quality",
        default="high_quality",
        choices=[*QUALITY_CLASSES, "all"],
        help="Quality class folder (default: high_quality).",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Override input image folder.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEBUG_OPENVEIN_DIR,
        help=f"Output root (default: {DEBUG_OPENVEIN_DIR.relative_to(PROJECT_ROOT)}).",
    )
    parser.add_argument(
        "--extractors",
        nargs="+",
        default=None,
        metavar="TAG",
        help=f"Extractors to run (default: all). Choices: {', '.join(EXTRACTOR_NAMES)}.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Process only the first N images (sorted by filename). Useful for smoke tests.",
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help=(
            "Remove stale feature PNGs in each extractor folder before running "
            "(files whose names do not match current inputs)."
        ),
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="If one extractor fails, log and continue with the remaining extractors.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate backend, paths, and extractors without running extraction.",
    )
    parser.add_argument(
        "--check-matlab-engine",
        action="store_true",
        help="Print MATLAB Engine import diagnostics and exit.",
    )
    parser.add_argument(
        "--list-extractors",
        action="store_true",
        help="Print extractor availability and exit.",
    )
    parser.add_argument(
        "--strict-preprocess",
        action="store_true",
        help="Python backend only: fail if Zhao09/Zhang09 are unavailable.",
    )
    parser.add_argument(
        "--skip-unavailable-extractors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Python backend only: skip unavailable extractors (default: true).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.list_extractors:
        _print_extractor_status()
        return 0

    if args.check_matlab_engine:
        from vascular_quality.openvein.matlab_backend import diagnose_matlab_engine

        return 0 if diagnose_matlab_engine() else 1

    if args.input is not None:
        if args.quality == "all":
            parser.error(
                "--quality all is not supported with --input; "
                "pick high_quality or low_quality."
            )
        qualities = [args.quality]
    else:
        if args.dataset is None:
            parser.error("--dataset is required unless --input is set.")
        qualities = list(iter_quality_classes(args.quality))

    toolkit_root = resolve_matlab_toolkit_root(
        str(args.matlab_toolkit_root) if args.matlab_toolkit_root else None
    )

    try:
        for q in qualities:
            if args.quality == "all" and len(qualities) > 1:
                print(f"\n{'=' * 60}\nQuality: {q}\n{'=' * 60}")
            run_extraction(
                backend=args.backend,
                dataset=args.dataset,
                quality=q,
                input_dir=args.input,
                output_root=args.output,
                extractors=args.extractors,
                matlab_toolkit_root=toolkit_root,
                continue_on_error=args.continue_on_error,
                dry_run=args.dry_run,
                limit=args.limit,
                skip_unavailable_preprocess=not args.strict_preprocess,
                skip_unavailable_extractors=args.skip_unavailable_extractors,
                clean_output=args.clean_output,
            )
    except (
        FileNotFoundError,
        ValueError,
        MatlabEngineNotInstalled,
        MatlabToolkitNotFound,
        MatlabExtractorError,
        ExtractorUnavailable,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not args.dry_run:
        print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
