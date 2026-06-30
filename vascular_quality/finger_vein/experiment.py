"""
Finger-vein PC extractor experiment (ISO Q1–Q9 + unified score).

Produces Excel/CSV results under results/finger_vein/PC/ by default.
"""

from __future__ import annotations

import argparse
import traceback
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from unified_quality import (
    COEFFICIENTS_ARE_PLACEHOLDER,
    unified_weight_description,
)
from vascular_quality.common.images import list_images_in_dir
from vascular_quality.common.openvein import vein_map_path
from vascular_quality.common.paths import (
    PROJECT_ROOT,
    ensure_dir,
    finger_vein_image_dir,
    iter_quality_classes,
    openvein_vein_map_dir,
)
from vascular_quality.common.debug_outputs import (
    finger_vein_image_debug_dir,
    remove_stale_debug_folders,
    validate_debug_layout,
)
from vascular_quality.finger_vein.config import (
    DEFAULT_CAPTURE_SITE,
    DEFAULT_VESSEL_CLEANUP_PRESET,
    FINGER_VEIN_DATASETS,
    OpenVeinVesselCleanupConfig,
    vessel_cleanup_from_preset,
)

from vascular_quality.common.pipeline import run_q1_q9_on_image

METRIC_MODALITY = "finger_vein"
DEFAULT_EXTRACTOR = "PC"

RESULT_COLUMNS: tuple[str, ...] = (
    "metric_modality",
    "dataset",
    "quality_folder",
    "extractor",
    "image_name",
    "vessel_cleanup",
    "Q1",
    "Q2",
    "Q3",
    "Q4",
    "Q5",
    "Q6",
    "Q7",
    "Q8",
    "Q9",
    "n_vessels",
    "endpoints",
    "intersections",
    "unified_score",
)

METRIC_COLS = [f"Q{i}" for i in range(1, 10)] + ["unified_score"]
SKELETON_COLS = ("n_vessels", "endpoints", "intersections")

PC_FEATURE_CMD = (
    'python -m vascular_quality.openvein.pipeline --backend matlab '
    '--matlab-toolkit-root "C:/Users/user/Downloads/OpenVein-Toolkit_v1.0.2" '
    '--dataset {dataset} --quality {quality} --extractors PC --clean-output'
)

PC_FEATURE_CMD_ALL = (
    'python -m vascular_quality.openvein.pipeline --backend matlab '
    '--matlab-toolkit-root "C:/Users/user/Downloads/OpenVein-Toolkit_v1.0.2" '
    '--dataset all --quality all --extractors PC --clean-output'
)


@dataclass
class ExperimentReport:
    success: bool = False
    datasets_processed: list[str] = field(default_factory=list)
    high_quality_images: int = 0
    low_quality_images: int = 0
    total_images: int = 0
    missing_pc_maps: list[str] = field(default_factory=list)
    rows_written: int = 0
    excel_file: Path | None = None
    csv_file: Path | None = None
    summary_file: Path | None = None
    log_file: Path | None = None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class DryRunReport:
    ok: bool
    log: str
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _normalize_extractor(name: str) -> str:
    ext = name.upper()
    if ext != "PC":
        raise ValueError(
            f"This experiment supports only extractor PC (MATLAB/OpenVein Principal Curvature). "
            f"Got {name!r}."
        )
    return ext


def _collect_jobs(
    datasets: list[str],
    qualities: list[str],
    extractor: str,
) -> list[tuple[str, str, Path, Path]]:
    jobs: list[tuple[str, str, Path, Path]] = []
    for dataset in datasets:
        if dataset not in FINGER_VEIN_DATASETS:
            raise ValueError(
                f"Unknown dataset {dataset!r}. Expected: {', '.join(FINGER_VEIN_DATASETS)}."
            )
        for quality in qualities:
            image_dir = finger_vein_image_dir(dataset, quality)
            if not image_dir.is_dir():
                continue
            vein_root = openvein_vein_map_dir(dataset, quality, extractor)
            for image_path in list_images_in_dir(image_dir):
                jobs.append((dataset, quality, image_path, vein_root))
    return jobs


def _row_from_result(
    dataset: str,
    quality: str,
    image_path: Path,
    result: dict[str, Any],
    extractor: str,
) -> dict[str, Any]:
    return {
        "metric_modality": METRIC_MODALITY,
        "dataset": dataset,
        "quality_folder": quality,
        "extractor": extractor,
        "image_name": image_path.name,
        "vessel_cleanup": result.get("vessel_cleanup", DEFAULT_VESSEL_CLEANUP_PRESET),
        "Q1": result["Q1"],
        "Q2": result["Q2"],
        "Q3": result["Q3"],
        "Q4": result["Q4"],
        "Q5": result["Q5"],
        "Q6": result["Q6"],
        "Q7": result["Q7"],
        "Q8": result["Q8"],
        "Q9": result["Q9"],
        "n_vessels": int(result["N_vessel"]),
        "endpoints": int(result["N_end"]),
        "intersections": int(result["N_int"]),
        "unified_score": result["unified_quality_score"],
    }


def _summary_by_dataset_quality(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (dataset, quality), group in df.groupby(["dataset", "quality_folder"], sort=True):
        row: dict[str, Any] = {
            "dataset": dataset,
            "quality_folder": quality,
            "extractor": group["extractor"].iloc[0],
            "n_images": len(group),
        }
        for col in METRIC_COLS:
            series = group[col].astype(float)
            row[f"{col}_mean"] = series.mean()
            row[f"{col}_std"] = series.std(ddof=0)
            row[f"{col}_min"] = series.min()
            row[f"{col}_max"] = series.max()
        rows.append(row)
    return pd.DataFrame(rows)


def _validate_result_df(df: pd.DataFrame, expected_rows: int) -> list[str]:
    warnings: list[str] = []
    if len(df) != expected_rows:
        warnings.append(f"Row count {len(df)} != expected {expected_rows}.")
    dupes = df.duplicated(subset=["dataset", "quality_folder", "image_name"]).sum()
    if dupes:
        warnings.append(f"{int(dupes)} duplicate image row(s).")
    for col in METRIC_COLS:
        if col not in df.columns:
            warnings.append(f"Missing column {col}.")
            continue
        if df[col].isna().any():
            warnings.append(f"NaN in {col}.")
        vals = df[col].astype(float)
        if not np.isfinite(vals).all():
            warnings.append(f"Non-finite values in {col}.")
        out = ((vals < 0) | (vals > 100)).sum()
        if out:
            warnings.append(f"{int(out)} value(s) in {col} outside [0, 100].")
    for col in SKELETON_COLS:
        if col not in df.columns:
            warnings.append(f"Missing column {col}.")
            continue
        if df[col].isna().any():
            warnings.append(f"NaN in {col}.")
        vals = df[col].astype(int)
        if (vals < 0).any():
            warnings.append(f"Negative value(s) in {col}.")
    return warnings


def run_dry_run(
    *,
    datasets: list[str],
    qualities: list[str],
    extractor: str,
    output_dir: Path,
    save_excel: bool,
) -> DryRunReport:
    extractor = _normalize_extractor(extractor)
    buf = StringIO()
    issues: list[str] = []
    warnings: list[str] = []

    buf.write("=" * 60 + "\n")
    buf.write("FINGER-VEIN PC EXPERIMENT DRY-RUN\n")
    buf.write("=" * 60 + "\n\n")

    total_inputs = 0
    total_pc_maps = 0
    missing_maps: list[str] = []
    extra_maps: list[str] = []

    for dataset in datasets:
        for quality in qualities:
            image_dir = finger_vein_image_dir(dataset, quality)
            vein_dir = openvein_vein_map_dir(dataset, quality, extractor)

            if not image_dir.is_dir():
                issues.append(f"Missing input folder: {image_dir}")
                continue

            images = list_images_in_dir(image_dir)
            total_inputs += len(images)
            buf.write(f"{dataset}/{quality}\n")
            buf.write(f"  Input images: {len(images)}\n")
            for img in images:
                buf.write(f"    - {img.name} ({img.suffix.lower()})\n")

            names = [p.name for p in images]
            dupes = [n for n, c in Counter(names).items() if c > 1]
            if dupes:
                issues.append(f"Duplicate filenames in {image_dir}: {', '.join(dupes)}")

            bmp_count = sum(1 for p in images if p.suffix.lower() == ".bmp")
            if bmp_count:
                buf.write(
                    f"  SCUT/BMP note: {bmp_count} .bmp input(s); OpenVein stages as "
                    f"{{stem}}.png before extraction.\n"
                )

            present = 0
            expected_stems = {p.stem for p in images}
            if vein_dir.is_dir():
                pc_files = sorted(vein_dir.glob("*.png"))
                pc_stems = {p.stem for p in pc_files}
                for img in images:
                    vpath = vein_map_path(vein_dir, img)
                    if vpath.is_file():
                        present += 1
                    else:
                        missing_maps.append(f"{dataset}/{quality}: {vpath.name}")
                extra = sorted(pc_stems - expected_stems)
                for stem in extra:
                    extra_maps.append(f"{dataset}/{quality}: {stem}.png")
            else:
                issues.append(f"Missing PC feature folder: {vein_dir}")
                for img in images:
                    missing_maps.append(
                        f"{dataset}/{quality}: {vein_map_path(vein_dir, img).name}"
                    )

            total_pc_maps += present
            buf.write(f"  PC feature maps: {present}/{len(images)} in {vein_dir}\n\n")

    buf.write(f"Total input images: {total_inputs}\n")
    buf.write(f"Total PC feature maps present: {total_pc_maps}\n\n")

    if missing_maps:
        issues.append(f"{len(missing_maps)} missing PC feature map(s):")
        for line in missing_maps[:8]:
            issues.append(f"  {line}")
        if len(missing_maps) > 8:
            issues.append(f"  ... +{len(missing_maps) - 8} more")
        buf.write("Generate missing PC maps (MATLAB/OpenVein):\n")
        affected = sorted({m.split(":")[0] for m in missing_maps})
        for key in affected:
            ds, q = key.split("/")
            buf.write(f"  {PC_FEATURE_CMD.format(dataset=ds, quality=q)}\n")
        buf.write(f"Or all datasets:\n  {PC_FEATURE_CMD_ALL}\n\n")

    if extra_maps:
        warnings.append(f"{len(extra_maps)} extra PC feature map(s) (stale outputs):")
        for line in extra_maps[:5]:
            warnings.append(f"  {line}")
        if len(extra_maps) > 5:
            warnings.append(f"  ... +{len(extra_maps) - 5} more")

    try:
        ensure_dir(output_dir)
        buf.write(f"Output directory writable: {output_dir}\n")
    except OSError as exc:
        issues.append(f"Cannot create output directory {output_dir}: {exc}")

    if save_excel:
        try:
            import openpyxl  # noqa: F401
            test_xlsx = output_dir / "_dry_run_excel_test.xlsx"
            pd.DataFrame({"test": [1]}).to_excel(test_xlsx, index=False, sheet_name="test")
            test_xlsx.unlink(missing_ok=True)
            buf.write("Excel writing: OK (openpyxl)\n")
        except ImportError:
            issues.append(
                "openpyxl is not installed. Run: pip install openpyxl"
            )
        except OSError as exc:
            issues.append(f"Excel write test failed: {exc}")

    ok = len(issues) == 0
    buf.write("\nISSUES:\n")
    if issues:
        for i in issues:
            buf.write(f"  - {i}\n")
    else:
        buf.write("  (none)\n")

    buf.write("\nWARNINGS:\n")
    if warnings:
        for w in warnings:
            buf.write(f"  - {w}\n")
    else:
        buf.write("  (none)\n")

    buf.write(f"\nDry-run {'PASSED' if ok else 'FAILED'}.\n")

    return DryRunReport(ok=ok, log=buf.getvalue(), issues=issues, warnings=warnings)


def _write_outputs(
    df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_dir: Path,
    *,
    save_excel: bool,
    log_text: str,
) -> tuple[Path | None, Path, Path | None, Path]:
    ensure_dir(output_dir)
    csv_path = output_dir / "q1_q9_pc_results.csv"
    log_path = output_dir / "q1_q9_pc_log.txt"
    df.to_csv(csv_path, index=False)

    excel_path: Path | None = None
    summary_xlsx: Path | None = None
    if save_excel:
        excel_path = output_dir / "q1_q9_pc_results.xlsx"
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="per_image_results", index=False)
            summary_df.to_excel(writer, sheet_name="summary_by_dataset_quality", index=False)
        summary_xlsx = output_dir / "q1_q9_pc_summary.xlsx"
        with pd.ExcelWriter(summary_xlsx, engine="openpyxl") as writer:
            summary_df.to_excel(writer, sheet_name="summary_by_dataset_quality", index=False)

    log_path.write_text(log_text, encoding="utf-8")
    return excel_path, csv_path, summary_xlsx, log_path


def run_experiment(
    *,
    datasets: list[str],
    qualities: list[str],
    extractor: str,
    output_dir: Path,
    save_excel: bool = True,
    save_debug_images: bool = False,
    limit: int | None = None,
    openvein_cleanup: OpenVeinVesselCleanupConfig | None = None,
) -> ExperimentReport:
    extractor = _normalize_extractor(extractor)
    cleanup = openvein_cleanup or vessel_cleanup_from_preset(DEFAULT_VESSEL_CLEANUP_PRESET)
    report = ExperimentReport()
    log = StringIO()
    log.write(f"Finger-vein PC experiment started: {datetime.now(timezone.utc).isoformat()}\n")
    log.write(f"Unified score: {unified_weight_description()}\n")
    log.write(f"COEFFICIENTS_ARE_PLACEHOLDER={COEFFICIENTS_ARE_PLACEHOLDER}\n")
    log.write(f"Vessel cleanup preset: {cleanup.preset_name}\n")
    log.write(f"Vessel cleanup config: {cleanup.summary()}\n")
    log.write(f"Save debug images: {save_debug_images}\n\n")

    dry = run_dry_run(
        datasets=datasets,
        qualities=qualities,
        extractor=extractor,
        output_dir=output_dir,
        save_excel=save_excel,
    )
    log.write(dry.log)
    if not dry.ok:
        report.errors = dry.issues
        report.warnings = dry.warnings
        _, csv_path, _, log_path = _write_outputs(
            pd.DataFrame(columns=RESULT_COLUMNS),
            pd.DataFrame(),
            output_dir,
            save_excel=False,
            log_text=log.getvalue() + "\nExperiment aborted: dry-run checks failed.\n",
        )
        report.csv_file = csv_path
        report.log_file = log_path
        return report

    jobs = _collect_jobs(datasets, qualities, extractor)
    if not jobs:
        report.errors.append("No runnable jobs (no input images found).")
        return report

    if limit is not None and limit > 0:
        jobs = jobs[:limit]
        log.write(f"Processing {len(jobs)} image(s) (--limit {limit}).\n\n")

    if save_debug_images:
        stale_removed = remove_stale_debug_folders(datasets, "all")
        log.write(f"Removed {len(stale_removed)} stale debug path(s).\n\n")

    rows: list[dict[str, Any]] = []
    runnable = 0
    for dataset, quality, image_path, vein_root in jobs:
        vein_path = vein_map_path(vein_root, image_path)
        if not vein_path.is_file():
            report.missing_pc_maps.append(str(vein_path))
            continue
        runnable += 1
        try:
            debug_dir = (
                finger_vein_image_debug_dir(dataset, quality, image_path.stem)
                if save_debug_images
                else None
            )
            result = run_q1_q9_on_image(
                image_path,
                vein_root,
                debug_dir,
                save_debug_images=save_debug_images,
                capture_site=DEFAULT_CAPTURE_SITE,
                openvein_cleanup=cleanup,
            )
            rows.append(
                _row_from_result(dataset, quality, image_path, result, extractor)
            )
            log.write(f"OK {dataset}/{quality}/{image_path.name}\n")
            if dataset not in report.datasets_processed:
                report.datasets_processed.append(dataset)
            if quality == "high_quality":
                report.high_quality_images += 1
            else:
                report.low_quality_images += 1
        except Exception as exc:
            msg = f"{dataset}/{quality}/{image_path.name}: {exc}"
            report.errors.append(msg)
            log.write(f"ERROR: {msg}\n{traceback.format_exc()}\n")

    report.total_images = report.high_quality_images + report.low_quality_images

    if not rows:
        report.errors.append("No results produced.")
        return report

    df = pd.DataFrame(rows)[list(RESULT_COLUMNS)]
    summary_df = _summary_by_dataset_quality(df)
    report.warnings.extend(_validate_result_df(df, runnable))
    report.rows_written = len(df)
    if save_debug_images:
        layout_issues = validate_debug_layout(datasets, "all")
        if layout_issues:
            for key, probs in layout_issues.items():
                report.warnings.append(f"Debug layout {key}: {'; '.join(probs)}")

    excel_path, csv_path, summary_xlsx, log_path = _write_outputs(
        df,
        summary_df,
        output_dir,
        save_excel=save_excel,
        log_text=log.getvalue(),
    )
    report.excel_file = excel_path
    report.csv_file = csv_path
    report.summary_file = summary_xlsx
    report.log_file = log_path
    report.success = len(report.errors) == 0 and report.rows_written == runnable
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Finger-vein ISO Q1–Q9 experiment using OpenVein PC (Principal Curvature) "
            "feature maps only."
        ),
    )
    parser.add_argument(
        "--extractor",
        default=DEFAULT_EXTRACTOR,
        help="OpenVein extractor (only PC is supported for this experiment).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(FINGER_VEIN_DATASETS),
        choices=list(FINGER_VEIN_DATASETS),
        help="Datasets to process (default: PLUS IDIAP SCUT).",
    )
    parser.add_argument(
        "--qualities",
        nargs="+",
        default=["high_quality", "low_quality"],
        choices=["high_quality", "low_quality"],
        help="Quality folders to process.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "finger_vein" / "PC",
        help="Output directory for CSV/Excel/log files.",
    )
    parser.add_argument(
        "--save-excel",
        action="store_true",
        help="Write Excel workbooks (q1_q9_pc_results.xlsx and q1_q9_pc_summary.xlsx).",
    )
    parser.add_argument(
        "--no-save-excel",
        action="store_true",
        help="Skip Excel output (CSV and log only).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and PC feature maps without computing metrics.",
    )
    parser.add_argument(
        "--vessel-cleanup",
        default=DEFAULT_VESSEL_CLEANUP_PRESET,
        choices=["iso_minimal", "heuristic_default"],
        help=(
            "Q8/Q9 OpenVein vein-map preprocessing preset. "
            "iso_minimal = ISO 5.2.8/5.2.9 steps b–c only (binarize + thin). "
            "heuristic_default = optional noise cleanup for PC maps (default; "
            "document in publications — not ISO-defined)."
        ),
    )
    parser.add_argument(
        "--save-debug-images",
        action="store_true",
        help=(
            "Write per-image debug PNGs under "
            "debug_outputs/finger_vein/{DATASET}/{quality}/{image_stem}/. "
            "Off by default for large experiments."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Process at most N images (smoke test; order: dataset, quality, filename).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    save_excel = args.save_excel and not args.no_save_excel

    try:
        _normalize_extractor(args.extractor)
    except ValueError as exc:
        parser.error(str(exc))

    if args.dry_run:
        report = run_dry_run(
            datasets=args.datasets,
            qualities=args.qualities,
            extractor=args.extractor,
            output_dir=args.output,
            save_excel=save_excel,
        )
        print(report.log)
        return 0 if report.ok else 1

    result = run_experiment(
        datasets=args.datasets,
        qualities=args.qualities,
        extractor=args.extractor,
        output_dir=args.output,
        save_excel=save_excel,
        save_debug_images=args.save_debug_images,
        limit=args.limit,
        openvein_cleanup=vessel_cleanup_from_preset(args.vessel_cleanup),
    )

    print(f"\nRows written: {result.rows_written}")
    if result.excel_file:
        print(f"Excel: {result.excel_file}")
    if result.csv_file:
        print(f"CSV: {result.csv_file}")
    if result.summary_file:
        print(f"Summary: {result.summary_file}")
    if result.warnings:
        print("Warnings:")
        for w in result.warnings:
            print(f"  - {w}")
    if result.errors:
        print("Errors:")
        for e in result.errors:
            print(f"  - {e}")

    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
