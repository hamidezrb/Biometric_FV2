"""
Configurable project paths for datasets and OpenVein debug features.

Layout (finger vein):
  data/finger_vein/{DATASET}/{quality}/images...
  debug_openvein_features/{DATASET}/{quality}/{EXTRACTOR}/images...
  debug_outputs/finger_vein/{DATASET}/{quality}/...
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

# Project root = parent of the vascular_quality package directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
DEBUG_OPENVEIN_DIR = PROJECT_ROOT / "debug_openvein_features"
DEBUG_OUTPUTS_DIR = PROJECT_ROOT / "debug_outputs"

# OpenVein vein-map tree (no anatomy prefix under debug_openvein_features):
#   debug_openvein_features/{DATASET}/{quality}/{EXTRACTOR}/*.png
#   DATASET ∈ PLUS, IDIAP, SCUT
#   quality ∈ high_quality, low_quality
#   EXTRACTOR ∈ RLT, MC, WLD, PC, GF, EMC
OPENVEIN_DATASETS: tuple[str, ...] = ("PLUS", "IDIAP", "SCUT")

QUALITY_CLASSES: tuple[str, ...] = ("high_quality", "low_quality")

OPENVEIN_EXTRACTORS: tuple[str, ...] = (
    "RLT",
    "MC",
    "WLD",
    "PC",
    "GF",
    "EMC",
)

DEFAULT_OPENVEIN_EXTRACTOR = "RLT"

IMAGE_EXTENSIONS: tuple[str, ...] = (
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
)


def finger_vein_root() -> Path:
    return DATA_DIR / "finger_vein"


def finger_vein_dataset_names() -> tuple[str, ...]:
    from vascular_quality.finger_vein.config import FINGER_VEIN_DATASETS

    return FINGER_VEIN_DATASETS


def finger_vein_image_dir(dataset: str, quality: str) -> Path:
    """data/finger_vein/{dataset}/{quality}/"""
    return finger_vein_root() / dataset / quality


def openvein_quality_dir(dataset: str, quality: str) -> Path:
    """debug_openvein_features/{dataset}/{quality}/"""
    return DEBUG_OPENVEIN_DIR / dataset / quality


def openvein_vein_map_dir(
    dataset: str,
    quality: str,
    extractor: str = DEFAULT_OPENVEIN_EXTRACTOR,
) -> Path:
    """debug_openvein_features/{dataset}/{quality}/{extractor}/"""
    return openvein_quality_dir(dataset, quality) / extractor


def finger_vein_vein_map_dir(
    dataset: str,
    quality: str,
    extractor: str = DEFAULT_OPENVEIN_EXTRACTOR,
) -> Path:
    """Alias for openvein_vein_map_dir (finger-vein datasets use the same tree)."""
    return openvein_vein_map_dir(dataset, quality, extractor)


def finger_vein_debug_output_dir(dataset: str, quality: str) -> Path:
    """debug_outputs/finger_vein/{dataset}/{quality}/"""
    return DEBUG_OUTPUTS_DIR / "finger_vein" / dataset / quality


def finger_vein_debug_run_dir(run_id: str) -> Path:
    """Timestamped debug visualization root: debug_outputs/finger_vein/_runs/{run_id}/"""
    return DEBUG_OUTPUTS_DIR / "finger_vein" / "_runs" / run_id


def palm_data_dir(quality: str) -> Path:
    """data/palm/{quality}/ — reserved for future palm experiments."""
    return DATA_DIR / "palm" / quality


def dorsal_hand_data_dir(quality: str) -> Path:
    """data/dorsal_hand/{quality}/ — reserved for future dorsal-hand experiments."""
    return DATA_DIR / "dorsal_hand" / quality


def ensure_dir(path: Path | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def iter_quality_classes(quality: str) -> Sequence[str]:
    """Expand 'all' to both quality folders."""
    if quality == "all":
        return QUALITY_CLASSES
    if quality not in QUALITY_CLASSES:
        raise ValueError(
            f"Unknown quality class {quality!r}. "
            f"Expected one of {QUALITY_CLASSES} or 'all'."
        )
    return (quality,)


def iter_dataset_classes(dataset: str) -> Sequence[str]:
    """Expand ``all`` to PLUS, IDIAP, SCUT (or accept a dataset name / path)."""
    if dataset == "all":
        return OPENVEIN_DATASETS
    path = Path(dataset)
    name = path.name if path.parts else dataset
    if name not in OPENVEIN_DATASETS:
        raise ValueError(
            f"Unknown dataset {name!r}. "
            f"Expected one of {OPENVEIN_DATASETS} or 'all'."
        )
    return (name,)


def resolve_image_paths(
    paths: Iterable[Path],
    *,
    label: str = "images",
) -> list[Path]:
    """Return existing image files; raise if the list is empty."""
    found = [p for p in paths if p.is_file()]
    if not found:
        searched = "\n  ".join(str(p.parent) for p in paths[:4])
        extra = f"\n  ... ({len(paths)} paths checked)" if len(paths) > 4 else ""
        raise FileNotFoundError(
            f"No {label} found.\n"
            f"Searched:\n  {searched}{extra}\n"
            f"Place vascular images under data/finger_vein/{{DATASET}}/{{quality}}/ "
            f"or pass --input explicitly."
        )
    return sorted(found)
