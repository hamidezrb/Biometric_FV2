"""Multi-extractor ISO Q1-Q9 evaluation and summary generation."""

from __future__ import annotations

import traceback
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tabulate import tabulate

from unified_quality import (
    COEFFICIENTS_ARE_PLACEHOLDER,
    DEFAULT_POWER_COEFFICIENTS,
    format_power_coefficients,
    unified_weight_description,
)
from vascular_quality.common.images import list_images_in_dir
from vascular_quality.common.openvein import vein_map_path
from vascular_quality.common.paths import (
    OPENVEIN_EXTRACTORS,
    PROJECT_ROOT,
    ensure_dir,
    finger_vein_image_dir,
    iter_quality_classes,
    openvein_vein_map_dir,
)
from vascular_quality.common.debug_outputs import (
    PRODUCTION_DEBUG_EXTRACTOR,
    finger_vein_image_debug_dir,
    remove_stale_debug_folders,
    validate_debug_layout,
)
from vascular_quality.finger_vein.config import (
    DEFAULT_CAPTURE_SITE,
    FINGER_VEIN_DATASETS,
)

from vascular_quality.common.pipeline import run_q1_q9_on_image

METRIC_COLUMNS = [f"Q{i}" for i in range(1, 10)] + ["Unified Score"]
SCORE_MIN, SCORE_MAX = 0, 100


def _results_to_dataframe(results: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append({
            "Image": r["file"],
            "Dataset": r["dataset"],
            "Quality": r["quality"],
            "Q1": r["Q1"],
            "Q2": r["Q2"],
            "Q3": r["Q3"],
            "Q4": r["Q4"],
            "Q5": r["Q5"],
            "Q6": r["Q6"],
            "Q7": r["Q7"],
            "Q8": r["Q8"],
            "Q9": r["Q9"],
            "Unified Score": r["unified_quality_score"],
            "Extractor": r.get("extractor", ""),
            "coefficients_are_placeholder": r.get("coefficients_are_placeholder", True),
            "N_vessel": r.get("N_vessel"),
            "N_end": r.get("N_end"),
            "N_int": r.get("N_int"),
            "N_fp": r.get("N_fp"),
        })
    return pd.DataFrame(rows)


def _dataset_summary(df: pd.DataFrame) -> pd.DataFrame:
    summaries = []
    for (dataset, quality), group in df.groupby(["Dataset", "Quality"], sort=True):
        row: dict[str, Any] = {
            "Dataset": dataset,
            "Quality": quality,
            "N_images": len(group),
        }
        for col in METRIC_COLUMNS:
            series = group[col].astype(float)
            row[f"{col}_mean"] = series.mean()
            row[f"{col}_median"] = series.median()
            row[f"{col}_std"] = series.std(ddof=0)
            row[f"{col}_min"] = series.min()
            row[f"{col}_max"] = series.max()
        unified = group["Unified Score"].astype(float)
        row["Unified_histogram"] = (
            f"min={unified.min():.4f}, med={unified.median():.4f}, max={unified.max():.4f}"
        )
        summaries.append(row)
    return pd.DataFrame(summaries)


def validate_metrics(
    df: pd.DataFrame,
    expected_rows: int,
    log: StringIO,
    *,
    extractor: str,
) -> list[str]:
    warnings: list[str] = []
    prefix = f"[{extractor}] " if extractor else ""

    if len(df) != expected_rows:
        warnings.append(
            f"{prefix}Row count ({len(df)}) != expected ({expected_rows})."
        )

    for col in METRIC_COLUMNS:
        if col not in df.columns:
            warnings.append(f"{prefix}Missing column {col}.")
            continue
        if df[col].isna().any():
            warnings.append(f"{prefix}NaN in {col}.")
        vals = df[col].astype(float)
        if not np.isfinite(vals).all():
            warnings.append(f"{prefix}Non-finite values in {col}.")
        out_of_range = ((vals < SCORE_MIN) | (vals > SCORE_MAX)).sum()
        if out_of_range:
            warnings.append(
                f"{prefix}{int(out_of_range)} value(s) in {col} outside [{SCORE_MIN}, {SCORE_MAX}]."
            )

    if not warnings:
        log.write(
            f"{prefix}VALIDATION OK: {len(df)} rows, all Q1-Q9 + Unified in range.\n"
        )
    return warnings


def _collect_jobs(
    datasets: list[str],
    quality: str,
    extractor: str,
) -> list[tuple[str, str, Path, Path]]:
    jobs: list[tuple[str, str, Path, Path]] = []
    for dataset in datasets:
        for q in iter_quality_classes(quality):
            images = list_images_in_dir(finger_vein_image_dir(dataset, q))
            if not images:
                continue
            vein_root = openvein_vein_map_dir(dataset, q, extractor)
            for image_path in images:
                if vein_map_path(vein_root, image_path).is_file():
                    jobs.append((dataset, q, image_path, vein_root))
    return jobs


def _extractor_comparison(all_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for extractor, group in all_df.groupby("Extractor", sort=True):
        unified = group["Unified Score"].astype(float)
        hq = group[group["Quality"] == "high_quality"]["Unified Score"].astype(float)
        lq = group[group["Quality"] == "low_quality"]["Unified Score"].astype(float)
        hq_mean = float(hq.mean()) if len(hq) else float("nan")
        lq_mean = float(lq.mean()) if len(lq) else float("nan")
        separation = hq_mean - lq_mean if np.isfinite(hq_mean) and np.isfinite(lq_mean) else float("nan")
        rows.append({
            "Extractor": extractor,
            "Mean Unified Score": unified.mean(),
            "Std": unified.std(ddof=0),
            "High Quality Mean": hq_mean,
            "Low Quality Mean": lq_mean,
            "HQ-LQ Separation": separation,
            "Images": len(group),
        })
    comp = pd.DataFrame(rows)
    if not comp.empty and comp["HQ-LQ Separation"].notna().any():
        # Prefer extractors where HQ mean > LQ mean; break ties by separation.
        valid = comp[comp["HQ-LQ Separation"].notna()]
        best_idx = valid["HQ-LQ Separation"].idxmax()
        comp["Best HQ/LQ Separator"] = False
        comp.loc[best_idx, "Best HQ/LQ Separator"] = True
    return comp


def _unified_score_statistics(all_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (extractor, dataset, quality), group in all_df.groupby(
        ["Extractor", "Dataset", "Quality"], sort=True
    ):
        u = group["Unified Score"].astype(float)
        rows.append({
            "Extractor": extractor,
            "Dataset": dataset,
            "Quality": quality,
            "N": len(group),
            "Mean": u.mean(),
            "Median": u.median(),
            "Std": u.std(ddof=0),
            "Min": u.min(),
            "Max": u.max(),
            "Q25": u.quantile(0.25),
            "Q75": u.quantile(0.75),
        })
    return pd.DataFrame(rows)


def run_multi_extractor_evaluation(
    *,
    datasets: list[str],
    quality: str = "all",
    extractors: list[str] | None = None,
    results_root: Path,
    save_debug_images: bool = False,
) -> tuple[pd.DataFrame, "ReadinessSummary"]:
    """Run Q1-Q9 for each extractor; write per-extractor and final_summary outputs."""
    extractors = extractors or list(OPENVEIN_EXTRACTORS)
    log = StringIO()
    log.write(f"Multi-extractor evaluation started: {datetime.now(timezone.utc).isoformat()}\n")
    log.write(f"Unified score weights: {unified_weight_description()}\n")
    log.write(f"alpha_i: {format_power_coefficients(DEFAULT_POWER_COEFFICIENTS)}\n")
    log.write(f"COEFFICIENTS_ARE_PLACEHOLDER={COEFFICIENTS_ARE_PLACEHOLDER}\n\n")

    stale_removed: list[str] = []
    if save_debug_images:
        stale_removed = remove_stale_debug_folders(datasets, quality)
        log.write(f"Removed {len(stale_removed)} stale debug path(s).\n\n")

    all_results: list[dict[str, Any]] = []
    all_warnings: list[str] = []
    all_errors: list[str] = []

    for extractor in extractors:
        log.write(f"\n{'=' * 40}\nEXTRACTOR: {extractor}\n{'=' * 40}\n")
        jobs = _collect_jobs(datasets, quality, extractor)
        log.write(f"Runnable jobs: {len(jobs)}\n")

        ext_results: list[dict[str, Any]] = []
        for dataset, q, image_path, vein_root in jobs:
            write_debug = (
                save_debug_images
                and extractor == PRODUCTION_DEBUG_EXTRACTOR
            )
            debug_dir = (
                finger_vein_image_debug_dir(dataset, q, image_path.stem)
                if write_debug
                else None
            )
            try:
                row = run_q1_q9_on_image(
                    image_path,
                    vein_root,
                    debug_dir,
                    save_debug_images=write_debug,
                    capture_site=DEFAULT_CAPTURE_SITE,
                )
                row["dataset"] = dataset
                row["quality"] = q
                row["extractor"] = extractor
                row["coefficients_are_placeholder"] = row.get(
                    "coefficients_are_placeholder",
                    COEFFICIENTS_ARE_PLACEHOLDER,
                )
                ext_results.append(row)
                all_results.append(row)
                log.write(f"OK {extractor} {dataset}/{q}/{image_path.name}\n")
            except Exception as exc:
                msg = f"{extractor} {dataset}/{q}/{image_path.name}: {exc}"
                all_errors.append(msg)
                log.write(f"ERROR: {msg}\n{traceback.format_exc()}\n")
                print(f"ERROR: {msg}")

        if not ext_results:
            all_warnings.append(f"No results for extractor {extractor}.")
            continue

        df = _results_to_dataframe(ext_results)
        all_warnings.extend(validate_metrics(df, len(jobs), log, extractor=extractor))

        ext_root = ensure_dir(results_root / extractor)
        csv_dir = ensure_dir(ext_root / "csv")
        tables_dir = ensure_dir(ext_root / "tables")
        df.to_csv(csv_dir / "per_image_results.csv", index=False)
        _dataset_summary(df).to_csv(csv_dir / "dataset_summary.csv", index=False)
        (tables_dir / "per_image_results.md").write_text(
            tabulate(df, headers="keys", tablefmt="github", floatfmt=".4f") + "\n",
            encoding="utf-8",
        )

    all_df = _results_to_dataframe(all_results) if all_results else pd.DataFrame()
    final_dir = ensure_dir(results_root / "final_summary")

    if save_debug_images:
        layout_issues = validate_debug_layout(datasets, quality)
        if layout_issues:
            for key, probs in layout_issues.items():
                all_warnings.append(f"Debug layout {key}: {'; '.join(probs)}")
        else:
            log.write("Debug layout validation: OK\n")

    if not all_df.empty:
        comp = _extractor_comparison(all_df)
        comp.to_csv(final_dir / "extractor_comparison.csv", index=False)
        _dataset_summary(all_df).to_csv(final_dir / "dataset_statistics.csv", index=False)
        _unified_score_statistics(all_df).to_csv(
            final_dir / "unified_score_statistics.csv", index=False
        )

        print("\nEXTRACTOR COMPARISON\n")
        print(tabulate(comp, headers="keys", tablefmt="github", floatfmt=".4f"))
        if comp["Best HQ/LQ Separator"].any():
            best = comp.loc[comp["Best HQ/LQ Separator"], "Extractor"].iloc[0]
            sep = comp.loc[comp["Best HQ/LQ Separator"], "HQ-LQ Separation"].iloc[0]
            print(
                f"\nBest high/low quality separator (HQ mean - LQ mean): "
                f"{best} (separation={sep:.2f})"
            )

    log_path = final_dir / "experiment_log.txt"
    log_path.write_text(log.getvalue(), encoding="utf-8")

    summary = ReadinessSummary(
        success=len(all_errors) == 0 and not all_df.empty,
        total_rows=len(all_df),
        errors=all_errors,
        warnings=all_warnings,
        files=[log_path],
    )
    if not all_df.empty:
        summary.files.extend([
            final_dir / "extractor_comparison.csv",
            final_dir / "dataset_statistics.csv",
            final_dir / "unified_score_statistics.csv",
        ])
    return all_df, summary


class ReadinessSummary:
    def __init__(
        self,
        *,
        success: bool,
        total_rows: int,
        errors: list[str],
        warnings: list[str],
        files: list[Path],
    ) -> None:
        self.success = success
        self.total_rows = total_rows
        self.errors = errors
        self.warnings = warnings
        self.files = files
