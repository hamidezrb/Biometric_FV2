"""
Shared experiment runner for non-finger vascular modalities (dorsal hand, palm).
"""

from __future__ import annotations

import argparse
import time
import traceback
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from io import StringIO
from itertools import groupby
from pathlib import Path
from typing import Any

import pandas as pd

from iso_constants import CaptureSite
from vascular_quality.common.images import list_images_in_dir
from vascular_quality.common.openvein import vein_map_path
from vascular_quality.common.paths import PROJECT_ROOT, ensure_dir, openvein_vein_map_dir
from vascular_quality.common.pipeline import run_q1_q9_on_image
from vascular_quality.finger_vein.config import (
    DEFAULT_VESSEL_CLEANUP_PRESET,
    OpenVeinVesselCleanupConfig,
    vessel_cleanup_from_preset,
)
from vascular_quality.finger_vein.experiment import (
    DEFAULT_EXTRACTOR,
    ExperimentProgress,
    ExperimentReport,
    RESULT_COLUMNS,
    _summary_by_dataset_quality,
    _validate_result_df,
    _write_outputs,
)


@dataclass(frozen=True)
class ModalitySpec:
    name: str
    data_root: Path
    capture_site: CaptureSite
    output_default: Path


def _modality_image_dir(spec: ModalitySpec, dataset: str, quality: str) -> Path:
    return spec.data_root / dataset / quality


def discover_supported_datasets(spec: ModalitySpec) -> list[str]:
    if not spec.data_root.is_dir():
        return []
    datasets: list[str] = []
    for child in sorted(spec.data_root.iterdir()):
        if not child.is_dir():
            continue
        has_quality_dir = any(q.is_dir() for q in child.iterdir())
        if has_quality_dir:
            datasets.append(child.name)
    return datasets


def _collect_jobs(
    spec: ModalitySpec,
    datasets: list[str],
    qualities: list[str],
    extractor: str,
) -> list[tuple[str, str, Path, Path]]:
    jobs: list[tuple[str, str, Path, Path]] = []
    supported = discover_supported_datasets(spec)
    for dataset in datasets:
        dataset_root = spec.data_root / dataset
        if not dataset_root.is_dir():
            raise ValueError(
                f"Unknown dataset {dataset!r} for modality {spec.name}. "
                f"Expected one of: {', '.join(supported) if supported else '(none detected)'}."
            )
        for quality in qualities:
            image_dir = _modality_image_dir(spec, dataset, quality)
            if not image_dir.is_dir():
                continue
            vein_root = openvein_vein_map_dir(dataset, quality, extractor)
            for image_path in list_images_in_dir(image_dir):
                jobs.append((dataset, quality, image_path, vein_root))
    return jobs


def _row_from_result(
    spec: ModalitySpec,
    dataset: str,
    quality: str,
    image_path: Path,
    result: dict[str, Any],
    extractor: str,
) -> dict[str, Any]:
    return {
        "metric_modality": spec.name,
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


def run_dry_run(
    *,
    spec: ModalitySpec,
    datasets: list[str],
    qualities: list[str],
    output_dir: Path,
    save_excel: bool,
) -> tuple[bool, str, list[str], list[str]]:
    buf = StringIO()
    issues: list[str] = []
    warnings: list[str] = []

    buf.write("=" * 60 + "\n")
    buf.write(f"{spec.name.upper()} EXPERIMENT DRY-RUN\n")
    buf.write("=" * 60 + "\n\n")

    total_inputs = 0
    total_pc_maps = 0
    missing_maps: list[str] = []
    extra_maps: list[str] = []

    for dataset in datasets:
        for quality in qualities:
            image_dir = _modality_image_dir(spec, dataset, quality)
            vein_dir = openvein_vein_map_dir(dataset, quality, DEFAULT_EXTRACTOR)

            if not image_dir.is_dir():
                issues.append(f"Missing input folder: {image_dir}")
                continue

            images = list_images_in_dir(image_dir)
            total_inputs += len(images)
            buf.write(f"{dataset}/{quality}\n")
            buf.write(f"  Input images: {len(images)}\n")

            names = [p.name for p in images]
            dupes = [n for n, c in Counter(names).items() if c > 1]
            if dupes:
                issues.append(f"Duplicate filenames in {image_dir}: {', '.join(dupes)}")

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
            issues.append("openpyxl is not installed. Run: pip install openpyxl")
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
    return ok, buf.getvalue(), issues, warnings


def run_modality_experiment(
    *,
    spec: ModalitySpec,
    datasets: list[str],
    qualities: list[str],
    output_dir: Path,
    save_excel: bool,
    save_csv: bool,
    no_debug: bool,
    progress: int,
    openvein_cleanup: OpenVeinVesselCleanupConfig | None = None,
) -> ExperimentReport:
    cleanup = openvein_cleanup or vessel_cleanup_from_preset(DEFAULT_VESSEL_CLEANUP_PRESET)
    save_debug_images = not no_debug
    report = ExperimentReport()
    log = StringIO()
    t0 = time.monotonic()
    log.write(f"{spec.name} experiment started: {datetime.now(timezone.utc).isoformat()}\n")
    log.write(f"Vessel cleanup preset: {cleanup.preset_name}\n")
    log.write(f"Save debug images: {save_debug_images}\n")
    log.write(f"Progress every: {max(1, progress)} image(s)\n\n")

    ok, dry_log, dry_issues, dry_warnings = run_dry_run(
        spec=spec,
        datasets=datasets,
        qualities=qualities,
        output_dir=output_dir,
        save_excel=save_excel,
    )
    log.write(dry_log)
    if not ok:
        report.errors = dry_issues
        report.warnings = dry_warnings
        report.elapsed_seconds = time.monotonic() - t0
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

    jobs = _collect_jobs(spec, datasets, qualities, DEFAULT_EXTRACTOR)
    if not jobs:
        report.errors.append("No runnable jobs (no input images found).")
        return report

    report.total_expected = len(jobs)
    progress_report = ExperimentProgress(
        log,
        total_jobs=report.total_expected,
        quiet=False,
        progress_every=max(1, progress),
    )
    progress_report.start_experiment(report.total_expected)

    rows: list[dict[str, Any]] = []
    runnable = 0

    try:
        for dataset, quality, group_jobs in _iter_job_groups(jobs):
            progress_report.start_group(
                dataset,
                quality,
                len(group_jobs),
                extractor=DEFAULT_EXTRACTOR,
                vessel_cleanup=cleanup.preset_name,
            )
            for _ds, _q, image_path, vein_root in group_jobs:
                vein_path = vein_map_path(vein_root, image_path)
                if not vein_path.is_file():
                    report.missing_pc_maps.append(str(vein_path))
                    report.total_skipped += 1
                    progress_report.warn_missing_pc_map(
                        dataset, quality, image_path.name, vein_path
                    )
                    log.write(
                        f"SKIP {dataset}/{quality}/{image_path.name} "
                        f"(missing PC map: {vein_path.name})\n"
                    )
                    continue

                runnable += 1
                try:
                    result = run_q1_q9_on_image(
                        image_path,
                        vein_root,
                        out_dir=(
                            output_dir
                            / "debug_outputs"
                            / dataset
                            / quality
                            / image_path.stem
                        ),
                        save_debug_images=save_debug_images,
                        capture_site=spec.capture_site,
                        openvein_cleanup=cleanup,
                    )
                    rows.append(
                        _row_from_result(
                            spec,
                            dataset,
                            quality,
                            image_path,
                            result,
                            DEFAULT_EXTRACTOR,
                        )
                    )
                    report.total_processed += 1
                    progress_report.on_image_done(dataset, quality, image_path.name)
                    print(
                        f"[{report.total_processed}/{report.total_expected}] "
                        f"{dataset} {quality} {image_path.name}"
                    )
                    print(
                        " ".join(
                            [
                                f"Q{i}={result[f'Q{i}']:.2f}"
                                for i in range(1, 10)
                            ]
                        )
                    )
                    if dataset not in report.datasets_processed:
                        report.datasets_processed.append(dataset)
                    if quality == "high_quality":
                        report.high_quality_images += 1
                    else:
                        report.low_quality_images += 1
                except Exception as exc:
                    report.total_failed += 1
                    msg = f"{dataset}/{quality}/{image_path.name}: {exc}"
                    report.errors.append(msg)
                    progress_report.warn_failure(dataset, quality, image_path.name, exc)
                    log.write(f"ERROR: {msg}\n{traceback.format_exc()}\n")
    finally:
        progress_report.close()

    report.total_images = report.high_quality_images + report.low_quality_images
    report.elapsed_seconds = time.monotonic() - t0

    if not rows:
        report.errors.append("No results produced.")
        report.csv_file = None
        report.log_file = None
        return report

    df = pd.DataFrame(rows)[list(RESULT_COLUMNS)]
    summary_df = _summary_by_dataset_quality(df)
    report.warnings.extend(_validate_result_df(df, runnable))
    report.rows_written = len(df)

    progress_report.write_summary(
        total_expected=report.total_expected,
        total_processed=report.total_processed,
        total_failed=report.total_failed,
        total_skipped=report.total_skipped,
        elapsed_seconds=report.elapsed_seconds,
        excel_file=output_dir / "q1_q9_pc_results.xlsx" if save_excel else None,
        csv_file=output_dir / "q1_q9_pc_results.csv" if save_csv else None,
        log_file=output_dir / "q1_q9_pc_log.txt",
    )

    excel_path = None
    csv_path = None
    summary_xlsx = None
    log_path = output_dir / "q1_q9_pc_log.txt"
    if save_csv:
        excel_path, csv_path, summary_xlsx, log_path = _write_outputs(
            df,
            summary_df,
            output_dir,
            save_excel=save_excel,
            log_text=log.getvalue(),
        )
    else:
        ensure_dir(output_dir)
        log_path.write_text(log.getvalue(), encoding="utf-8")
        if save_excel:
            excel_path = output_dir / "q1_q9_pc_results.xlsx"
            with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
                df.to_excel(writer, sheet_name="per_image_results", index=False)
                summary_df.to_excel(
                    writer, sheet_name="summary_by_dataset_quality", index=False
                )
            summary_xlsx = output_dir / "q1_q9_pc_summary.xlsx"
            with pd.ExcelWriter(summary_xlsx, engine="openpyxl") as writer:
                summary_df.to_excel(
                    writer, sheet_name="summary_by_dataset_quality", index=False
                )

    report.excel_file = excel_path
    report.csv_file = csv_path
    report.summary_file = summary_xlsx
    report.log_file = log_path
    report.success = len(report.errors) == 0 and report.rows_written == runnable
    return report


def build_modality_arg_parser(
    *,
    modality_name: str,
    output_default: Path,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"{modality_name} ISO Q1–Q9 experiment using OpenVein PC feature maps.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help=f"Datasets to process (default: auto-detect under data/{modality_name}).",
    )
    parser.add_argument(
        "--qualities",
        nargs="+",
        default=["high_quality", "low_quality"],
        help="Quality folders to process.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=output_default,
        help="Output directory for CSV/Excel/log files.",
    )
    parser.add_argument(
        "--save-excel",
        action="store_true",
        help="Write Excel workbooks.",
    )
    parser.add_argument(
        "--save-csv",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write CSV results (default: true).",
    )
    parser.add_argument(
        "--no-debug",
        action="store_true",
        help="Disable debug image generation (default behavior).",
    )
    parser.add_argument(
        "--progress",
        type=int,
        default=1,
        metavar="N",
        help="Print progress every N processed images (default: 1).",
    )
    return parser


def _iter_job_groups(
    jobs: list[tuple[str, str, Path, Path]],
) -> list[tuple[str, str, list[tuple[str, str, Path, Path]]]]:
    groups: list[tuple[str, str, list[tuple[str, str, Path, Path]]]] = []
    for (dataset, quality), group in groupby(jobs, key=lambda j: (j[0], j[1])):
        groups.append((dataset, quality, list(group)))
    return groups
