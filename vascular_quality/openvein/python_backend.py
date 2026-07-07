"""
Python fallback backend — approximate OpenVein extractors (not identical).

Use only for quick local experiments. For research / ISO evaluation, prefer
``--backend matlab`` which calls the original OpenVein toolkit.
"""

from __future__ import annotations

from pathlib import Path

from vascular_quality.common.paths import openvein_quality_dir
from vascular_quality.openvein.backend import (
    BackendKind,
    ExtractionBackend,
    ExtractionJob,
)
from vascular_quality.openvein.extractors import (
    ExtractorParity,
    ExtractorUnavailable,
    extract_feature,
    get_extractor,
    list_extractors,
)
from vascular_quality.openvein.io import (
    collect_written_outputs,
    extractor_output_dir,
    list_input_images,
    load_grayscale,
    remove_stale_outputs,
    report_stale_outputs,
    save_feature_image,
    validate_input_output_mapping,
)
from vascular_quality.openvein.preprocessing import preprocess_openvein


class PythonOpenVeinBackend(ExtractionBackend):
    """Approximate OpenVein extraction in pure Python (experimental)."""

    @property
    def kind(self) -> BackendKind:
        return "python"

    def validate_runtime(self, *, toolkit_root: Path | None = None) -> None:  # noqa: ARG002
        return

    def _dry_run_backend(
        self,
        job: ExtractionJob,
        *,
        toolkit_root: Path | None = None,  # noqa: ARG002
    ) -> None:
        print("Python backend: APPROXIMATE — not OpenVein-identical.")
        for spec in list_extractors():
            if spec.tag not in job.extractors:
                continue
            note = f" — {spec.notes}" if spec.notes else ""
            print(f"  {spec.tag}: {spec.parity.value}{note}")

    def run(
        self,
        job: ExtractionJob,
        *,
        toolkit_root: Path | None = None,  # noqa: ARG002
        continue_on_error: bool = False,
        skip_unavailable_preprocess: bool = True,
        skip_unavailable_extractors: bool = True,
        clean_output: bool = False,
    ) -> dict[str, list[Path]]:
        images = list(job.image_paths) if job.image_paths else list_input_images(job.image_dir)
        if job.limit is not None:
            images = images[: job.limit]
        written: dict[str, list[Path]] = {tag: [] for tag in job.extractors}

        print("Backend: python (APPROXIMATE — not for final experiments)")
        print(f"Modality: {job.modality}")
        limit_note = f", limit={job.limit}" if job.limit is not None else ""
        print(f"Input:   {job.image_dir} ({len(images)} image(s){limit_note})")
        print(
            f"Output:  {openvein_quality_dir(job.dataset, job.quality, modality=job.modality, output_root=job.output_root)}/{{EXTRACTOR}}/"
        )
        print(f"Preprocess skip_unavailable={skip_unavailable_preprocess}")

        for tag in job.extractors:
            spec = get_extractor(tag)
            if spec.parity == ExtractorParity.UNAVAILABLE:
                msg = (
                    f"Extractor {tag} ({spec.full_name}) is not available in Python: "
                    f"{spec.notes or 'use --backend matlab for original OpenVein.'}"
                )
                if skip_unavailable_extractors:
                    print(f"[skip] {msg}")
                    continue
                if continue_on_error:
                    print(f"ERROR: {msg}")
                    continue
                raise ExtractorUnavailable(msg)

            out_dir = extractor_output_dir(
                job.output_root, job.modality, job.dataset, job.quality, tag
            )
            if clean_output:
                remove_stale_outputs(out_dir, images, extractor=tag)
            else:
                report_stale_outputs(out_dir, images, extractor=tag)

            print(f"\n--- {tag} ({spec.full_name}) [{spec.parity.value}] ---")
            if spec.notes:
                print(f"    Note: {spec.notes}")

            try:
                for image_path in images:
                    gray = load_grayscale(image_path)
                    preprocessed = preprocess_openvein(
                        gray,
                        skip_unavailable=skip_unavailable_preprocess,
                    )
                    feature = extract_feature(preprocessed, tag)
                    out_path = out_dir / image_path.name
                    save_feature_image(feature, out_path)
                    written[tag].append(out_path)
                written[tag] = collect_written_outputs(out_dir, images)
                validate_input_output_mapping(images, out_dir, extractor=tag)
                print(f"Saved {len(written[tag])} feature image(s) to: {out_dir}")
            except Exception as exc:
                msg = f"Extractor {tag} failed: {exc}"
                print(f"ERROR: {msg}")
                if not continue_on_error:
                    raise
        return written
