"""Full-hand ISO Q1-Q9 experiment runner."""

from __future__ import annotations

from vascular_quality.common.modality_experiment import (
    ModalitySpec,
    build_modality_arg_parser,
    discover_supported_datasets,
    run_modality_experiment,
)
from vascular_quality.common.paths import PROJECT_ROOT
from vascular_quality.full_hand.config import DEFAULT_CAPTURE_SITE


def main(argv: list[str] | None = None) -> int:
    spec = ModalitySpec(
        name="full_hand",
        data_root=PROJECT_ROOT / "data" / "full_hand",
        capture_site=DEFAULT_CAPTURE_SITE,
        output_default=PROJECT_ROOT / "results" / "full_hand",
    )
    parser = build_modality_arg_parser(
        modality_name=spec.name,
        output_default=spec.output_default,
    )
    args = parser.parse_args(argv)

    datasets = args.datasets or discover_supported_datasets(spec)
    if not datasets:
        parser.error(
            "No full-hand datasets detected under data/full_hand/. "
            "Pass --datasets explicitly after creating dataset folders."
        )

    report = run_modality_experiment(
        spec=spec,
        datasets=datasets,
        qualities=args.qualities,
        output_dir=args.output,
        save_excel=args.save_excel,
        save_csv=args.save_csv,
        no_debug=args.no_debug,
        progress=args.progress,
    )

    if report.warnings:
        print("\nWarnings:")
        for w in report.warnings:
            print(f"  - {w}")
    if report.errors:
        print("\nErrors:")
        for e in report.errors:
            print(f"  - {e}")
    return 0 if report.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
