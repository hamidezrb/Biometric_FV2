"""
OpenVein extraction backend protocol and shared job validation.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from vascular_quality.common.paths import (
    DEBUG_OPENVEIN_DIR,
    DEFAULT_MODALITY,
    OPENVEIN_EXTRACTORS,
    QUALITY_CLASSES,
    export_dataset_name,
    iter_modality_dataset_classes,
    modality_image_dir,
    openvein_quality_dir,
)
from vascular_quality.finger_vein.config import FINGER_VEIN_DATASETS
from vascular_quality.openvein.extractors import EXTRACTOR_NAMES
from vascular_quality.openvein.io import extractor_output_dir, list_input_images, report_stale_outputs

BackendKind = Literal["matlab", "python"]

DEFAULT_BACKEND: BackendKind = "matlab"
DEFAULT_MATLAB_TOOLKIT_ENV = "OPENVEIN_TOOLKIT_ROOT"


@dataclass(frozen=True)
class ExtractionJob:
    """One modality / dataset / quality extraction run."""

    modality: str
    dataset: str
    quality: str
    image_dir: Path
    output_root: Path
    extractors: tuple[str, ...] = field(default_factory=lambda: EXTRACTOR_NAMES)
    image_paths: tuple[Path, ...] = ()
    limit: int | None = None


def parse_dataset_name(dataset_arg: str, *, modality: str = DEFAULT_MODALITY) -> str:
    """Accept a dataset name or a path like ``data/finger_vein/PLUS``."""
    path = Path(dataset_arg)
    name = path.name if path.parts else dataset_arg
    if modality == "finger_vein":
        if name not in FINGER_VEIN_DATASETS:
            raise ValueError(
                f"Unknown dataset {name!r}. Expected one of: {', '.join(FINGER_VEIN_DATASETS)}."
            )
        return name
    resolved = tuple(iter_modality_dataset_classes(modality, name))
    return resolved[0]


def resolve_extraction_job(
    *,
    modality: str = DEFAULT_MODALITY,
    dataset: str | None,
    quality: str,
    input_dir: Path | None = None,
    output_root: Path | str | None = None,
    extractors: list[str] | None = None,
    limit: int | None = None,
) -> ExtractionJob:
    """Build a validated extraction job from CLI-style arguments."""
    if input_dir is not None:
        folder = Path(input_dir)
        if not folder.is_dir():
            raise FileNotFoundError(f"Input folder does not exist: {folder}")
        ds = dataset or "CUSTOM"
        q = quality
    else:
        if dataset is None:
            raise ValueError(
                f"Provide --dataset for modality {modality!r} "
                "or --input pointing at an image folder."
            )
        ds = parse_dataset_name(dataset, modality=modality)
        if quality not in QUALITY_CLASSES:
            raise ValueError(
                f"Unknown quality {quality!r}. Expected: {', '.join(QUALITY_CLASSES)}."
            )
        folder = modality_image_dir(modality, ds, quality)
        if not folder.is_dir():
            raise FileNotFoundError(
                f"Input folder missing: {folder}\n"
                f"Expected layout: data/{modality}/{{DATASET}}/{{quality}}/"
            )
        q = quality

    tags = tuple(e.upper() for e in (extractors or list(EXTRACTOR_NAMES)))
    for tag in tags:
        if tag not in OPENVEIN_EXTRACTORS:
            raise ValueError(
                f"Unknown extractor {tag!r}. Expected one of: {', '.join(EXTRACTOR_NAMES)}."
            )

    out_root = Path(output_root) if output_root else DEBUG_OPENVEIN_DIR
    images = list_input_images(folder)
    if limit is not None:
        if limit < 1:
            raise ValueError(f"--limit must be >= 1, got {limit}.")
        images = images[:limit]

    return ExtractionJob(
        modality=modality,
        dataset=ds,
        quality=q,
        image_dir=folder,
        output_root=out_root,
        extractors=tags,
        image_paths=tuple(images),
        limit=limit,
    )


def resolve_matlab_toolkit_root(cli_value: str | None) -> Path | None:
    """Resolve OpenVein toolkit path from CLI or OPENVEIN_TOOLKIT_ROOT."""
    raw = cli_value or os.environ.get(DEFAULT_MATLAB_TOOLKIT_ENV)
    if not raw:
        return None
    return Path(raw).expanduser().resolve()


class ExtractionBackend(ABC):
    """Backend that runs OpenVein-style feature extraction."""

    @property
    @abstractmethod
    def kind(self) -> BackendKind:
        """``matlab`` or ``python``."""

    @abstractmethod
    def validate_runtime(self, *, toolkit_root: Path | None = None) -> None:
        """Raise with a clear message when the backend cannot run."""

    def validate_job(self, job: ExtractionJob) -> list[Path]:
        """Common path validation; returns sorted input images (respecting --limit)."""
        if job.image_paths:
            return list(job.image_paths)
        images = list_input_images(job.image_dir)
        if job.limit is not None:
            return images[: job.limit]
        return images

    def dry_run(self, job: ExtractionJob, *, toolkit_root: Path | None = None) -> None:
        """Validate paths, backend, and extractors without extracting."""
        self.validate_runtime(toolkit_root=toolkit_root, require_engine=False)
        images = self.validate_job(job)
        print(f"Backend:     {self.kind}")
        print(f"Modality:    {job.modality}")
        ds_label = export_dataset_name(job.dataset)
        if ds_label:
            print(f"Dataset:     {ds_label}")
        print(f"Quality:     {job.quality}")
        limit_note = f" (limit {job.limit})" if job.limit is not None else ""
        print(f"Input:       {job.image_dir} ({len(images)} image(s){limit_note})")
        print(
            f"Output root: {openvein_quality_dir(job.dataset, job.quality, modality=job.modality, output_root=job.output_root)}/"
        )
        print(f"Extractors:  {', '.join(job.extractors)}")
        for tag in job.extractors:
            out_dir = extractor_output_dir(
                job.output_root, job.modality, job.dataset, job.quality, tag
            )
            print(f"  {tag} -> {out_dir}")
        for tag in job.extractors:
            out_dir = extractor_output_dir(
                job.output_root, job.modality, job.dataset, job.quality, tag
            )
            report_stale_outputs(out_dir, images, extractor=tag)
        self._dry_run_backend(job, toolkit_root=toolkit_root)
        if self.kind == "matlab":
            from vascular_quality.openvein.matlab_backend import (
                MatlabEngineNotInstalled,
                _import_matlab_engine,
            )

            try:
                _import_matlab_engine()
                print("MATLAB Engine: installed")
            except MatlabEngineNotInstalled as exc:
                print("MATLAB Engine: NOT INSTALLED (required for real extraction)")
                print(str(exc))

    @abstractmethod
    def _dry_run_backend(
        self,
        job: ExtractionJob,
        *,
        toolkit_root: Path | None = None,
    ) -> None:
        """Backend-specific dry-run checks."""

    @abstractmethod
    def run(
        self,
        job: ExtractionJob,
        *,
        toolkit_root: Path | None = None,
        continue_on_error: bool = False,
        skip_unavailable_preprocess: bool = True,
        skip_unavailable_extractors: bool = True,
        clean_output: bool = False,
    ) -> dict[str, list[Path]]:
        """Run extraction; return extractor tag -> written file paths."""


def get_backend(kind: BackendKind) -> ExtractionBackend:
    if kind == "matlab":
        from vascular_quality.openvein.matlab_backend import MatlabOpenVeinBackend

        return MatlabOpenVeinBackend()
    if kind == "python":
        from vascular_quality.openvein.python_backend import PythonOpenVeinBackend

        return PythonOpenVeinBackend()
    raise ValueError(f"Unknown backend {kind!r}. Expected 'matlab' or 'python'.")
