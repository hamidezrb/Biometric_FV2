"""
MATLAB Engine backend — original OpenVein toolkit via ``matlab.engine``.

This is the recommended backend for research experiments because it runs the
same Matcher / FeatureType algorithms as the OpenVein MATLAB toolkit.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

from vascular_quality.openvein.backend import (
    DEFAULT_MATLAB_TOOLKIT_ENV,
    BackendKind,
    ExtractionBackend,
    ExtractionJob,
)
from vascular_quality.common.paths import openvein_quality_dir
from vascular_quality.openvein.io import (
    collect_written_outputs,
    extractor_output_dir,
    list_input_images,
    prepare_matlab_input_dir,
    remove_stale_outputs,
    report_stale_outputs,
    validate_input_output_mapping,
)

_OPENVEIN_MATLAB_DIR = Path(__file__).resolve().parent

MATLAB_ENGINE_HELP = textwrap.dedent(
    """
    MATLAB Engine for Python is not installed or could not be imported.

    Install (from an elevated shell if needed):
      1. Install MATLAB (same major release as your Engine build).
      2. cd "%MATLABROOT%\\extern\\engines\\python"
      3. python setup.py install

    Verify in Python:
      import matlab.engine
      eng = matlab.engine.start_matlab()

    Docs:
      https://www.mathworks.com/help/matlab/matlab-engine-for-python.html

    Optional: set OPENVEIN_TOOLKIT_ROOT to your OpenVein-Toolkit install path.
    """
).strip()


class MatlabEngineNotInstalled(RuntimeError):
    """Raised when ``import matlab.engine`` fails."""


class MatlabToolkitNotFound(FileNotFoundError):
    """Raised when the OpenVein toolkit directory is missing."""


class MatlabExtractorError(RuntimeError):
    """Raised when one OpenVein extractor fails under MATLAB."""


def _import_matlab_engine():
    """
    Import the MATLAB Engine module.

    Returns ``matlab.engine`` (the module). Call ``.start_matlab()`` on it —
    not ``.engine.start_matlab()``.
    """
    try:
        import matlab.engine as matlab_engine  # type: ignore[import-untyped]
    except ImportError as exc:
        raise MatlabEngineNotInstalled(MATLAB_ENGINE_HELP) from exc

    if not hasattr(matlab_engine, "start_matlab"):
        raise MatlabEngineNotInstalled(
            "matlab.engine imported but start_matlab is missing.\n"
            f"  matlab.engine = {matlab_engine!r}\n"
            f"  dir sample: {[a for a in dir(matlab_engine) if not a.startswith('_')][:10]}\n"
            "Reinstall from %MATLABROOT%\\extern\\engines\\python (python setup.py install)."
        )
    return matlab_engine


def diagnose_matlab_engine() -> bool:
    """
    Print MATLAB Engine import diagnostics.

    Returns True when ``matlab.engine.start_matlab`` is available.
    """
    try:
        import matlab.engine as matlab_engine  # type: ignore[import-untyped]
    except ImportError as exc:
        print(f"import matlab.engine: FAILED ({exc})")
        return False

    print(f"matlab.engine: {matlab_engine}")
    ok = hasattr(matlab_engine, "start_matlab")
    print(f"hasattr(matlab.engine, 'start_matlab'): {ok}")
    return ok


def _matlab_path(path: Path) -> str:
    """Windows-safe absolute path for MATLAB Engine calls."""
    return str(path.resolve())


class MatlabOpenVeinBackend(ExtractionBackend):
    """Run OpenVein via MATLAB Engine (original algorithms)."""

    def __init__(self) -> None:
        self._eng = None

    @property
    def kind(self) -> BackendKind:
        return "matlab"

    def validate_runtime(
        self,
        *,
        toolkit_root: Path | None = None,
        require_engine: bool = True,
    ) -> None:
        if toolkit_root is None:
            raise ValueError(
                "MATLAB backend requires --matlab-toolkit-root or "
                f"{DEFAULT_MATLAB_TOOLKIT_ENV} environment variable."
            )
        if not toolkit_root.is_dir():
            raise MatlabToolkitNotFound(
                f"OpenVein toolkit folder not found: {toolkit_root}\n"
                f"Pass --matlab-toolkit-root pointing at OpenVein-Toolkit_vX.Y.Z"
            )
        if require_engine:
            _import_matlab_engine()

    def _dry_run_backend(
        self,
        job: ExtractionJob,
        *,
        toolkit_root: Path | None = None,
    ) -> None:
        self.validate_runtime(toolkit_root=toolkit_root, require_engine=False)
        assert toolkit_root is not None
        m_file = _OPENVEIN_MATLAB_DIR / "openvein_matlab_extract_one.m"
        if not m_file.is_file():
            raise FileNotFoundError(f"MATLAB helper missing: {m_file}")
        print(f"Toolkit:     {toolkit_root}")
        print(f"MATLAB hook: {m_file.name}")
        print("Dry-run OK — paths and toolkit validated.")

    def _start_engine(self, toolkit_root: Path):
        matlab_engine = _import_matlab_engine()
        if self._eng is None:
            print("Starting MATLAB Engine...")
            self._eng = matlab_engine.start_matlab()
            self._eng.addpath(_matlab_path(_OPENVEIN_MATLAB_DIR), nargout=0)
            self._eng.addpath(
                self._eng.genpath(_matlab_path(toolkit_root)),
                nargout=0,
            )
        return self._eng

    def close(self) -> None:
        if self._eng is not None:
            print("Stopping MATLAB Engine...")
            self._eng.quit()
            self._eng = None

    def run(
        self,
        job: ExtractionJob,
        *,
        toolkit_root: Path | None = None,
        continue_on_error: bool = False,
        skip_unavailable_preprocess: bool = True,  # noqa: ARG002 — MATLAB only
        skip_unavailable_extractors: bool = True,  # noqa: ARG002 — MATLAB only
        clean_output: bool = False,
    ) -> dict[str, list[Path]]:
        self.validate_runtime(toolkit_root=toolkit_root)
        assert toolkit_root is not None

        images = list(job.image_paths) if job.image_paths else list_input_images(job.image_dir)
        if job.limit is not None:
            images = images[: job.limit]
        written: dict[str, list[Path]] = {tag: [] for tag in job.extractors}
        errors: list[str] = []

        print(f"Backend: matlab (original OpenVein)")
        print(f"Modality: {job.modality}")
        limit_note = f", limit={job.limit}" if job.limit is not None else ""
        print(f"Input:   {job.image_dir} ({len(images)} image(s){limit_note})")
        print(
            f"Output:  {openvein_quality_dir(job.dataset, job.quality, modality=job.modality, output_root=job.output_root)}/{{EXTRACTOR}}/"
        )
        print(f"Toolkit: {toolkit_root}")

        input_dir, cleanup_dir = prepare_matlab_input_dir(job.image_dir, images)
        try:
            eng = self._start_engine(toolkit_root)
            for tag in job.extractors:
                out_dir = extractor_output_dir(
                    job.output_root, job.modality, job.dataset, job.quality, tag
                )
                if clean_output:
                    remove_stale_outputs(out_dir, images, extractor=tag)
                else:
                    report_stale_outputs(out_dir, images, extractor=tag)

                print(f"\n--- {tag} (OpenVein / MATLAB) ---")
                try:
                    count = eng.openvein_matlab_extract_one(
                        _matlab_path(input_dir),
                        _matlab_path(out_dir),
                        tag,
                        nargout=1,
                    )
                    count = int(count)
                    if count != len(images):
                        print(
                            f"WARNING: MATLAB saved {count} file(s) but "
                            f"{len(images)} input image(s) were expected."
                        )
                    written[tag] = collect_written_outputs(out_dir, images)
                    validate_input_output_mapping(
                        images, out_dir, extractor=tag
                    )
                    print(f"Saved {len(written[tag])} feature image(s) to: {out_dir}")
                except Exception as exc:  # MATLAB Engine wraps errors broadly
                    msg = f"Extractor {tag} failed: {exc}"
                    print(f"ERROR: {msg}")
                    errors.append(msg)
                    if not continue_on_error:
                        raise MatlabExtractorError(msg) from exc
        finally:
            if cleanup_dir is not None:
                import shutil

                shutil.rmtree(cleanup_dir, ignore_errors=True)
            self.close()

        if errors and continue_on_error:
            print(f"\nCompleted with {len(errors)} extractor error(s).")
        return written


if __name__ == "__main__":
    raise SystemExit(0 if diagnose_matlab_engine() else 1)
