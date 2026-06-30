"""
OpenVein-style vascular feature extraction.

Recommended flow (research / ISO experiments)::

    Dataset -> Python pipeline -> MATLAB Engine -> OpenVein toolkit -> feature images

Fallback::

    Dataset -> Python pipeline -> python backend (approximate, experimental)
"""

from vascular_quality.openvein.backend import (
    DEFAULT_BACKEND,
    DEFAULT_MATLAB_TOOLKIT_ENV,
    BackendKind,
    ExtractionJob,
    get_backend,
)
from vascular_quality.openvein.extractors import (
    EXTRACTOR_NAMES,
    ExtractorSpec,
    extract_feature,
    get_extractor,
    list_extractors,
)
from vascular_quality.openvein.preprocessing import preprocess_openvein

__all__ = [
    "DEFAULT_BACKEND",
    "DEFAULT_MATLAB_TOOLKIT_ENV",
    "BackendKind",
    "EXTRACTOR_NAMES",
    "ExtractionJob",
    "ExtractorSpec",
    "extract_feature",
    "get_backend",
    "get_extractor",
    "list_extractors",
    "preprocess_openvein",
    "run_extraction",
]


def run_extraction(*args, **kwargs):
    """Batch feature extraction — see ``vascular_quality.openvein.pipeline``."""
    from vascular_quality.openvein.pipeline import run_extraction as _run

    return _run(*args, **kwargs)
