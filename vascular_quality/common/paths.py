"""
Configurable project paths for datasets and OpenVein debug features.

Input layout:
  data/{modality}/{DATASET}/{quality}/   (Layout A)
  data/{modality}/{quality}/             (Layout B, dorsal/palm flat)

OpenVein feature-map output layout:
  debug_openvein_features/finger_vein/{DATASET}/{quality}/{EXTRACTOR}/
  debug_openvein_features/{modality}/{quality}/{EXTRACTOR}/          (flat)
  debug_openvein_features/{modality}/{DATASET}/{quality}/{EXTRACTOR}/ (named datasets)
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
#   Finger vein DATASET ∈ PLUS, IDIAP, SCUT; dorsal/palm datasets are auto-discovered
#   quality ∈ high_quality, low_quality
#   EXTRACTOR ∈ RLT, MC, WLD, PC, GF, EMC
OPENVEIN_DATASETS: tuple[str, ...] = ("PLUS", "IDIAP", "SCUT")

VASCULAR_MODALITIES: tuple[str, ...] = ("finger_vein", "dorsal_hand", "palm")
DEFAULT_MODALITY = "finger_vein"

# Virtual dataset token for Layout B (internal only — never written to output paths or exports).
DEFAULT_FLAT_DATASET = "default"

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


def modality_data_root(modality: str = DEFAULT_MODALITY) -> Path:
    """data/{modality}/"""
    if modality not in VASCULAR_MODALITIES:
        raise ValueError(
            f"Unknown modality {modality!r}. Expected one of: {', '.join(VASCULAR_MODALITIES)}."
        )
    return DATA_DIR / modality


def modality_has_flat_quality_layout(modality: str = DEFAULT_MODALITY) -> bool:
    """True when quality folders exist directly under data/{modality}/ (Layout B)."""
    root = modality_data_root(modality)
    return root.is_dir() and any((root / q).is_dir() for q in QUALITY_CLASSES)


def modality_image_dir(modality: str, dataset: str, quality: str) -> Path:
    """
    Resolve the image folder for a modality / dataset / quality triple.

    Layout A: data/{modality}/{dataset}/{quality}/
    Layout B: data/{modality}/{quality}/  (dataset must be DEFAULT_FLAT_DATASET)
    """
    root = modality_data_root(modality)
    if dataset == DEFAULT_FLAT_DATASET:
        return root / quality
    return root / dataset / quality


def discover_modality_datasets(modality: str = DEFAULT_MODALITY) -> tuple[str, ...]:
    """
    Return dataset names under data/{modality}/.

    Layout A: one name per subfolder containing quality folders
    (e.g. Bosphorus, NCUT).
    Layout B: returns (DEFAULT_FLAT_DATASET,) when high_quality/low_quality
    exist directly under data/{modality}/.
    Both layouts may coexist.
    """
    root = modality_data_root(modality)
    if not root.is_dir():
        return ()
    datasets: list[str] = []
    if modality_has_flat_quality_layout(modality):
        datasets.append(DEFAULT_FLAT_DATASET)
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name in QUALITY_CLASSES:
            continue
        if any((child / q).is_dir() for q in QUALITY_CLASSES):
            datasets.append(child.name)
    return tuple(datasets)


def finger_vein_root() -> Path:
    return modality_data_root("finger_vein")


def finger_vein_dataset_names() -> tuple[str, ...]:
    from vascular_quality.finger_vein.config import FINGER_VEIN_DATASETS

    return FINGER_VEIN_DATASETS


def finger_vein_image_dir(dataset: str, quality: str) -> Path:
    """data/finger_vein/{dataset}/{quality}/"""
    return modality_image_dir("finger_vein", dataset, quality)


def export_dataset_name(dataset: str) -> str:
    """Dataset label for CSV/Excel/logs (empty for internal flat-layout token)."""
    return "" if dataset == DEFAULT_FLAT_DATASET else dataset


def format_known_datasets(discovered: Sequence[str]) -> str:
    """User-facing dataset list; flat layout shown as '(flat layout)', not the internal token."""
    visible = [d for d in discovered if d != DEFAULT_FLAT_DATASET]
    if DEFAULT_FLAT_DATASET in discovered:
        visible.append("(flat layout)")
    return ", ".join(visible) if visible else "(none detected)"


def progress_group_label(modality: str, dataset: str, quality: str) -> str:
    """Progress/log group key without exposing the internal flat-layout token."""
    if dataset == DEFAULT_FLAT_DATASET:
        return f"{modality}/{quality}"
    return f"{dataset}/{quality}"


def openvein_quality_dir(
    dataset: str,
    quality: str,
    *,
    modality: str = DEFAULT_MODALITY,
    output_root: Path | None = None,
) -> Path:
    """
    OpenVein feature-map parent directory (quality level, no extractor).

    Finger vein: {root}/finger_vein/{dataset}/{quality}/
    Flat dorsal/palm: {root}/{modality}/{quality}/
    Named dorsal/palm: {root}/{modality}/{dataset}/{quality}/
    """
    root = Path(output_root) if output_root is not None else DEBUG_OPENVEIN_DIR
    if modality == "finger_vein":
        return root / "finger_vein" / dataset / quality
    if dataset == DEFAULT_FLAT_DATASET:
        return root / modality / quality
    return root / modality / dataset / quality


def openvein_vein_map_dir(
    dataset: str,
    quality: str,
    extractor: str = DEFAULT_OPENVEIN_EXTRACTOR,
    *,
    modality: str = DEFAULT_MODALITY,
    output_root: Path | None = None,
) -> Path:
    """OpenVein extractor output directory for one modality/dataset/quality run."""
    return openvein_quality_dir(
        dataset,
        quality,
        modality=modality,
        output_root=output_root,
    ) / extractor


def finger_vein_vein_map_dir(
    dataset: str,
    quality: str,
    extractor: str = DEFAULT_OPENVEIN_EXTRACTOR,
) -> Path:
    """debug_openvein_features/finger_vein/{dataset}/{quality}/{extractor}/"""
    return openvein_vein_map_dir(
        dataset,
        quality,
        extractor,
        modality="finger_vein",
    )


def finger_vein_debug_output_dir(dataset: str, quality: str) -> Path:
    """debug_outputs/finger_vein/{dataset}/{quality}/"""
    return DEBUG_OUTPUTS_DIR / "finger_vein" / dataset / quality


def finger_vein_debug_run_dir(run_id: str) -> Path:
    """Timestamped debug visualization root: debug_outputs/finger_vein/_runs/{run_id}/"""
    return DEBUG_OUTPUTS_DIR / "finger_vein" / "_runs" / run_id


def palm_data_dir(quality: str) -> Path:
    """Legacy helper — prefer modality_image_dir('palm', dataset, quality)."""
    return DATA_DIR / "palm" / quality


def dorsal_hand_data_dir(quality: str) -> Path:
    """Legacy helper — prefer modality_image_dir('dorsal_hand', dataset, quality)."""
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


def iter_modality_dataset_classes(
    modality: str,
    dataset: str | None,
) -> Sequence[str]:
    """
    Resolve dataset names for OpenVein extraction.

    Finger vein: ``dataset`` is required (PLUS, IDIAP, SCUT, or all).
    Dorsal hand / palm: ``dataset`` may be omitted or ``all`` to auto-detect
    every folder under ``data/{modality}/``.
    """
    if modality not in VASCULAR_MODALITIES:
        raise ValueError(
            f"Unknown modality {modality!r}. "
            f"Expected one of: {', '.join(VASCULAR_MODALITIES)}."
        )

    if modality == "finger_vein":
        if dataset is None:
            raise ValueError(
                "Provide --dataset (PLUS, IDIAP, SCUT, or all) "
                "or --input pointing at an image folder."
            )
        return iter_dataset_classes(dataset)

    discovered = discover_modality_datasets(modality)
    if dataset is None or dataset == "all":
        if not discovered:
            raise ValueError(
                f"No datasets found under {modality_data_root(modality)}. "
                f"Create data/{modality}/{{DATASET}}/{{high_quality|low_quality}}/ "
                f"or data/{modality}/{{high_quality|low_quality}}/ first."
            )
        return discovered

    path = Path(dataset)
    name = path.name if path.parts else dataset
    if name == DEFAULT_FLAT_DATASET:
        if not modality_has_flat_quality_layout(modality):
            known = format_known_datasets(discovered)
            raise ValueError(
                f"Flat layout not found for modality {modality}. "
                f"Expected quality folders under data/{modality}/. "
                f"Known datasets: {known}."
            )
        return (DEFAULT_FLAT_DATASET,)

    ds_root = modality_data_root(modality) / name
    if not ds_root.is_dir() or name not in discovered:
        known = format_known_datasets(discovered)
        raise ValueError(
            f"Unknown dataset {name!r} for modality {modality}. "
            f"Expected one of: {known}, 'all', or a path under data/{modality}/."
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
