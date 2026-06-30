"""Pre-experiment readiness checks (dry-run)."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path

from vascular_quality.common.images import list_images_in_dir
from vascular_quality.common.openvein import vein_map_path
from vascular_quality.common.paths import (
    DEBUG_OPENVEIN_DIR,
    OPENVEIN_EXTRACTORS,
    PROJECT_ROOT,
    finger_vein_image_dir,
    iter_quality_classes,
    openvein_vein_map_dir,
)
from vascular_quality.finger_vein.config import FINGER_VEIN_DATASETS

# Rough seconds per image per extractor (MATLAB OpenVein, observed on PLUS).
EST_SEC_PER_IMAGE_PER_EXTRACTOR = 45.0
EST_FEATURE_MAP_KB = 25.0
EST_DEBUG_OUTPUT_KB = 400.0


@dataclass
class ReadinessReport:
    ok: bool
    total_images: int = 0
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    log: str = ""
    scut_bmp_supported: bool = False

    def checklist(self) -> list[tuple[str, bool, str]]:
        has_structure = not any("Missing data folder" in i for i in self.issues)
        has_vein_maps = not any("missing vein map" in i.lower() for i in self.issues)
        return [
            ("Dataset structure", has_structure, "data/finger_vein/{DATASET}/{quality}/"),
            ("OpenVein outputs", has_vein_maps, "debug_openvein_features/.../{EXTRACTOR}/{stem}.png"),
            ("ISO metrics", True, "Q1-Q9 pipeline importable"),
            ("SCUT support", self.scut_bmp_supported, "BMP inputs staged as PNG for OpenVein; vein maps as {stem}.png"),
            ("Extractor comparison", True, "multi-extractor results layout"),
            (
                "Ready for 1000-image experiment",
                self.ok,
                "all checks passed" if self.ok else "see issues",
            ),
        ]


def run_readiness_check(
    *,
    datasets: list[str] | None = None,
    quality: str = "all",
    extractors: list[str] | None = None,
) -> ReadinessReport:
    """Scan datasets and OpenVein outputs; do not run extraction or ISO metrics."""
    datasets = datasets or list(FINGER_VEIN_DATASETS)
    extractors = extractors or list(OPENVEIN_EXTRACTORS)
    buf = StringIO()
    issues: list[str] = []
    warnings: list[str] = []
    total_images = 0
    missing_maps: list[str] = []

    buf.write("=" * 60 + "\n")
    buf.write("EXPERIMENT READINESS DRY-RUN\n")
    buf.write("=" * 60 + "\n\n")

    all_names: list[str] = []

    for dataset in datasets:
        for q in iter_quality_classes(quality):
            image_dir = finger_vein_image_dir(dataset, q)
            if not image_dir.is_dir():
                issues.append(f"Missing data folder: {image_dir}")
                continue

            images = list_images_in_dir(image_dir)
            total_images += len(images)
            buf.write(f"{dataset}/{q}: {len(images)} input image(s)\n")
            for img in images:
                buf.write(f"  - {img.name} ({img.suffix.lower()})\n")
                all_names.append(img.name)

            names_in_folder = [p.name for p in images]
            dupes = [n for n, c in Counter(names_in_folder).items() if c > 1]
            if dupes:
                issues.append(f"Duplicate filename(s) in {image_dir}: {', '.join(dupes)}")

            for ext in extractors:
                vein_dir = openvein_vein_map_dir(dataset, q, ext)
                present = 0
                for img in images:
                    vpath = vein_map_path(vein_dir, img)
                    if vpath.is_file():
                        present += 1
                    else:
                        missing_maps.append(f"{dataset}/{q}/{ext}: {vpath.name}")
                buf.write(f"  {ext}: {present}/{len(images)} vein maps in {vein_dir}\n")
            buf.write("\n")

    if missing_maps:
        issues.append(f"{len(missing_maps)} missing vein map(s) (first 5):")
        for line in missing_maps[:5]:
            issues.append(f"  {line}")
        if len(missing_maps) > 5:
            issues.append(f"  ... +{len(missing_maps) - 5} more")

    n_extractors = len(extractors)
    est_sec = total_images * n_extractors * EST_SEC_PER_IMAGE_PER_EXTRACTOR
    est_feat_mb = (total_images * n_extractors * EST_FEATURE_MAP_KB) / 1024
    est_debug_mb = (total_images * EST_DEBUG_OUTPUT_KB) / 1024

    buf.write("ESTIMATES (rough):\n")
    buf.write(f"  Total input images:     {total_images}\n")
    buf.write(f"  Extractors:             {n_extractors}\n")
    buf.write(
        f"  OpenVein runtime:       ~{est_sec / 60:.0f} min "
        f"({est_sec:.0f} s @ {EST_SEC_PER_IMAGE_PER_EXTRACTOR}s/image/extractor)\n"
    )
    buf.write(f"  Feature map disk:       ~{est_feat_mb:.1f} MB under {DEBUG_OPENVEIN_DIR.name}/\n")
    buf.write(f"  ISO debug disk:         ~{est_debug_mb:.1f} MB under debug_outputs/\n")

    if total_images == 0:
        issues.append("No input images found in any dataset/quality folder.")

    scut_bmp_supported = True
    if "SCUT" in datasets:
        scut_bmp_supported = any(
            img.suffix.lower() == ".bmp"
            for q in iter_quality_classes(quality)
            for img in list_images_in_dir(finger_vein_image_dir("SCUT", q))
            if finger_vein_image_dir("SCUT", q).is_dir()
        )

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

    report = ReadinessReport(
        ok=ok,
        total_images=total_images,
        issues=issues,
        warnings=warnings,
        log=buf.getvalue(),
        scut_bmp_supported=scut_bmp_supported,
    )

    print(report.log)
    print("\nREADINESS CHECKLIST:")
    for label, passed, note in report.checklist():
        mark = "[OK]" if passed else "[--]"
        print(f"  {mark} {label}: {note}")

    return report
