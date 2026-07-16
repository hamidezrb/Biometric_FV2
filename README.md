# ISO/IEC 29794-9 Vascular Image Quality Assessment

Python implementation of ISO/IEC 29794-9 quality components (Q1–Q9) for vascular biometrics, with OpenVein feature extraction via the MATLAB Engine.

## Supported modalities

| Modality | Experiment script | Capture site | Package |
|----------|-------------------|--------------|---------|
| **Finger Vein** | `run_finger_vein_experiment.py` | Finger second phalanx | `vascular_quality/finger_vein/` |
| **Palm Vein** | `run_palm_experiment.py` | Palm / dorsal ROI | `vascular_quality/palm/` |
| **Dorsal Hand Vein** | `run_dorsal_hand_experiment.py` | Palm / dorsal ROI | `vascular_quality/dorsal_hand/` |
| **Full Hand** | `run_fullhand_experiment.py` | Full hand (incl. fingers) | `vascular_quality/full_hand/` |

## Install

```powershell
pip install numpy opencv-python scikit-image pandas openpyxl
```

Optional (total progress bar during large experiments):

```powershell
pip install tqdm
```

## Pipeline overview

```text
data/{modality}/…                          ← input images (not in Git)
        ↓
vascular_quality.openvein.pipeline         ← OpenVein PC extraction (MATLAB)
        ↓
debug_openvein_features/{modality}/…       ← vein maps (not in Git)
        ↓
run_*_experiment.py                        ← ISO Q1–Q9 + unified score
        ↓
results/{modality}/                        ← Excel / CSV / log
```

### Recommended workflow (production)

1. **Extract PC features** (MATLAB) for all datasets and qualities.
2. **Dry-run** the quality experiment — verify image and PC map counts match.
3. **Smoke test** with `--limit 30` on a subset (finger vein) or a small quality folder.
4. **Full experiment** with progress reporting (debug images off by default).

---

## Capture-site coefficients (ISO/IEC 29794-9)

Q1 (effective area), Q8 (total vascular length), and Q9 (feature points) use capture-site-dependent normalizers **Sc**, **Lc**, and **Fc** (also called **Pc** in some draft text). Values are defined in `iso_constants.py` and selected by each modality’s `DEFAULT_CAPTURE_SITE`.

| Capture site | Used by | Sc | Lc | Fc (Pc) |
|--------------|---------|----|----|---------|
| Finger second phalanx | Finger Vein | 20,000 | 600 | 15 |
| Palm or dorsal ROI | Palm Vein, Dorsal Hand Vein | 40,000 | 1,200 | 25 |
| Full hand | Full Hand | 300,000 | 9,000 | 50 |

**Full Hand** corresponds to the entire hand including fingers. It must use the Full Hand coefficients above — not the Palm/Dorsal ROI values.

Normalization coefficients for Q1–Q9 are therefore **capture-site dependent**. Palm/Dorsal ROI and Full Hand share the same metric pipeline but apply different Sc/Lc/Fc according to the ISO/IEC 29794-9 draft tables.

---

## Dataset structure

Place images under:

```text
data/
  finger_vein/
    PLUS/
      high_quality/
      low_quality/
    IDIAP/
      high_quality/
      low_quality/
    SCUT/
      high_quality/
      low_quality/
  dorsal_hand/
    {DATASET}/              ← Layout A (named datasets, e.g. Bosphorus)
      high_quality/
      low_quality/
    high_quality/           ← Layout B (flat; no dataset subfolder)
    low_quality/
  palm/
    {DATASET}/              ← Layout A
      high_quality/
      low_quality/
    high_quality/           ← Layout B (flat; no dataset subfolder)
    low_quality/
  full_hand/
    {DATASET}/              ← Layout A (named datasets)
      high_quality/
      low_quality/
    high_quality/           ← Layout B (flat; no dataset subfolder)
    low_quality/
```

### Layout notes

- **Finger vein** always uses Layout A with fixed dataset names (`PLUS`, `IDIAP`, `SCUT`).
- **Palm**, **dorsal hand**, and **full hand** support **two folder layouts** (both can coexist):
  - **Layout A:** `data/{modality}/{DATASET}/{quality}/` — dataset names are auto-detected.
  - **Layout B:** `data/{modality}/{quality}/` — images live directly under the modality folder (no dataset subfolder).

Supported formats: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff`

**Git:** folder structure is tracked via `.gitkeep` files; **image files are ignored** and must be added locally on each machine.

Optional one-time setup for finger-vein folders (creates folders + migrates legacy layouts):

```powershell
python scripts/setup_finger_vein_layout.py
```

---

## MATLAB Engine for Python

### Why it is needed

OpenVein’s original algorithms (RLT, MC, WLD, **PC**, GF, EMC) ship as a MATLAB toolkit. The **MATLAB backend** (`--backend matlab`) runs those algorithms through `matlab.engine` — required for Principal Curvature (PC) and publication-grade feature maps.

The **Python backend** (`--backend python`) is an approximate fallback (partial RLT/GF only). **Do not use it for final experiments** unless you explicitly document that choice.

### Install

1. Install MATLAB (same release as your OpenVein toolkit).
2. Install the engine API:

```powershell
cd "$env:MATLABROOT\extern\engines\python"
python setup.py install
```

3. Test installation:

```powershell
python -c "import matlab.engine; print('MATLAB Engine OK')"
```

4. (Optional) Set toolkit path:

```powershell
$env:OPENVEIN_TOOLKIT_ROOT = "C:\Users\user\Downloads\OpenVein-Toolkit_v1.0.2"
```

Docs: [MATLAB Engine for Python](https://www.mathworks.com/help/matlab/matlab-engine-for-python.html)

### OpenVein Feature Extraction (MATLAB Backend)

OpenVein feature extraction supports these vascular modalities:

- **Finger Vein** (`--modality finger_vein`, default)
- **Dorsal Hand** (`--modality dorsal_hand`)
- **Palm** (`--modality palm`)
- **Full Hand** (`--modality full_hand`)

All modalities use the same MATLAB backend, extractor interfaces, output layout, logging, and CLI flags. Feature maps are written under `debug_openvein_features/` using modality-prefixed paths (see [Output layout](#output-layout) below).

Required toolkit:

```text
OpenVein-Toolkit_v1.0.2
```

#### Folder structure

**Finger vein** (Layout A only):

```text
data/finger_vein/{PLUS|IDIAP|SCUT}/{high_quality|low_quality}/
```

**Dorsal hand / palm / full hand** — Layout A (named datasets):

```text
data/{modality}/{DATASET}/{high_quality|low_quality}/
```

**Dorsal hand / palm / full hand** — Layout B (flat, no dataset subfolder):

```text
data/{modality}/{high_quality|low_quality}/
```

Both layouts may coexist under the same modality.

Full tree example:

```text
data/
  finger_vein/
    PLUS/
      high_quality/
      low_quality/
  dorsal_hand/
    high_quality/           ← Layout B
    low_quality/
    Bosphorus/              ← Layout A
      high_quality/
      low_quality/
  palm/
    high_quality/           ← Layout B
    low_quality/
    NCUT/                   ← Layout A
      high_quality/
      low_quality/
  full_hand/
    high_quality/           ← Layout B
    low_quality/
```

#### Output layout

```text
debug_openvein_features/
  finger_vein/
    PLUS/
      high_quality/
        PC/
          {stem}.png
  dorsal_hand/
    high_quality/           ← flat layout (Layout B)
      PC/
    Bosphorus/              ← named dataset (Layout A)
      high_quality/
        PC/
  palm/
    high_quality/
      PC/
  full_hand/
    high_quality/
      PC/
```

Dataset names for dorsal hand, palm, and full hand are **auto-detected** from folders under `data/{modality}/`. Flat layouts (quality folders directly under the modality) do not create a dataset segment in output paths.

#### Example commands

Finger Vein (single dataset / quality):

```powershell
python -m vascular_quality.openvein.pipeline --backend matlab --matlab-toolkit-root "C:/Users/user/Downloads/OpenVein-Toolkit_v1.0.2" --dataset PLUS --quality high_quality --extractors PC --output debug_openvein_features --continue-on-error
```

Finger Vein (all datasets and qualities):

```powershell
python -m vascular_quality.openvein.pipeline --backend matlab --matlab-toolkit-root "C:/Users/user/Downloads/OpenVein-Toolkit_v1.0.2" --dataset all --quality all --extractors PC --output debug_openvein_features --clean-output --continue-on-error
```

Dorsal Hand (all auto-detected datasets):

```powershell
python -m vascular_quality.openvein.pipeline --modality dorsal_hand --backend matlab --matlab-toolkit-root "C:/Users/user/Downloads/OpenVein-Toolkit_v1.0.2" --quality high_quality --extractors PC --output debug_openvein_features --continue-on-error
```

Palm (all auto-detected datasets):

```powershell
python -m vascular_quality.openvein.pipeline --modality palm --backend matlab --matlab-toolkit-root "C:/Users/user/Downloads/OpenVein-Toolkit_v1.0.2" --quality high_quality --extractors PC --output debug_openvein_features --continue-on-error
```

Full Hand (all auto-detected datasets):

```powershell
python -m vascular_quality.openvein.pipeline --modality full_hand --backend matlab --matlab-toolkit-root "C:/Users/user/Downloads/OpenVein-Toolkit_v1.0.2" --quality high_quality --extractors PC --output debug_openvein_features --continue-on-error
```

Dry-run (validate paths and counts, no extraction):

```powershell
python -m vascular_quality.openvein.pipeline --backend matlab --matlab-toolkit-root "C:/Users/user/Downloads/OpenVein-Toolkit_v1.0.2" --dataset all --quality all --extractors PC --dry-run
```

| Flag | Purpose |
|------|---------|
| `--modality` | `finger_vein` (default), `dorsal_hand`, `palm`, or `full_hand` |
| `--dataset` | Dataset name, path, or `all`. Required for finger vein unless `--input` is set. Optional for dorsal hand / palm / full hand (auto-detect when omitted). |
| `--dataset all` | Finger vein: PLUS, IDIAP, SCUT. Other modalities: every dataset under `data/{modality}/`. |
| `--quality all` | Run `high_quality` and `low_quality` |
| `--continue-on-error` | Log a failed extractor and continue |
| `--clean-output` | Remove stale vein maps before extraction |
| `--limit N` | Process only the first N images (smoke test) |
| `--backend python` | Approximate fallback only (not for final experiments) |

#### Modality-specific notes

- Finger vein behavior is unchanged when `--modality` is omitted (defaults to `finger_vein`).
- Dorsal hand, palm, and full hand use the same OpenVein MATLAB algorithms; only input paths differ.
- Flat layouts write feature maps to `debug_openvein_features/{modality}/{quality}/` with no dataset folder segment.
- Use distinct dataset folder names across modalities if sharing one `--output` root.

---

## Main experiment scripts

### `run_finger_vein_experiment.py` — production PC experiment

| | |
|---|---|
| **Purpose** | Compute Q1–Q9 + unified score using **PC (Principal Curvature)** OpenVein maps; export Excel/CSV for statistical analysis |
| **When to use** | Final experiments and thesis analysis (single extractor: PC) |
| **Required inputs** | Images in `data/finger_vein/{DATASET}/{quality}/`; PC maps in `debug_openvein_features/finger_vein/{DATASET}/{quality}/PC/{stem}.png` |
| **Output** | `q1_q9_pc_results.xlsx`, `q1_q9_pc_results.csv`, `q1_q9_pc_summary.xlsx`, `q1_q9_pc_log.txt` |
| **Capture site** | Finger second phalanx (Sc=20000, Lc=600, Fc=15) |

**Defaults for large runs:** debug images **off**, `heuristic_default` vessel cleanup, CSV + log always written; Excel when `--save-excel` is set.

#### Commands

Dry-run (check image and PC map counts):

```powershell
python run_finger_vein_experiment.py --extractor PC --datasets PLUS IDIAP SCUT --qualities high_quality low_quality --output results/finger_vein/PC --save-excel --vessel-cleanup heuristic_default --dry-run
```

Smoke test (first N images):

```powershell
python run_finger_vein_experiment.py --extractor PC --datasets PLUS IDIAP SCUT --qualities high_quality low_quality --output results/finger_vein/PC_test --save-excel --vessel-cleanup heuristic_default --limit 30 --progress-every 5
```

Full experiment (1000+ images):

```powershell
python run_finger_vein_experiment.py --extractor PC --datasets PLUS IDIAP SCUT --qualities high_quality low_quality --output results/finger_vein/PC --save-excel --vessel-cleanup heuristic_default --progress-every 10
```

Optional per-image debug PNGs (slow; large disk use):

```powershell
python run_finger_vein_experiment.py --extractor PC --datasets PLUS --qualities high_quality --output results/finger_vein/PC --save-excel --vessel-cleanup heuristic_default --save-debug-images
```

#### Progress reporting

During processing the terminal shows group headers, per-image progress with ETA, and a final summary. Example:

```text
Starting group: PLUS / high_quality
Images in group: 61
Extractor: PC
Vessel cleanup: heuristic_default

[PLUS/high_quality] 10/61 processed | current: image.png | elapsed: 00:00:32 | ETA: 00:01:04
Total progress: 10/203 images processed (4.9%)

Experiment finished
Total images expected: 203
Total processed: 203
Total failed: 0
Total skipped: 0
Elapsed time: 02:15:30
```

The same lines are written to `q1_q9_pc_log.txt`. Install `tqdm` for an additional total progress bar.

| Flag | Purpose |
|------|---------|
| `--vessel-cleanup` | `heuristic_default` (default) or `iso_minimal` — see [Q8/Q9 skeleton](#iso-compliance-q8--q9-vessel-skeleton) |
| `--progress-every N` | Print group/total progress every N processed images (default: 1) |
| `--quiet` | Errors, skips, and final summary only |
| `--save-debug-images` | Write debug PNGs under `debug_outputs/finger_vein/` (off by default) |
| `--limit N` | Process at most N images (smoke test) |
| `--save-excel` | Write Excel workbooks |
| `--dry-run` | Validate inputs without computing metrics |

#### Export columns (no file paths)

`metric_modality`, `dataset`, `quality_folder`, `extractor`, `image_name`, `vessel_cleanup`, `Q1`–`Q9`, `n_vessels`, `endpoints`, `intersections`, `unified_score`

| Column | Meaning |
|--------|---------|
| `n_vessels` | Skeleton pixel count in foreground **R** used for Q8 |
| `endpoints` | Endpoint count in **R** used for Q9 |
| `intersections` | Intersection count in **R** used for Q9 |
| `vessel_cleanup` | `heuristic_default` or `iso_minimal` (logged for reproducibility) |

---

### `run_palm_experiment.py` / `run_dorsal_hand_experiment.py`

Palm and dorsal hand ROI experiments reuse the shared modality runner (`vascular_quality.common.modality_experiment`) with capture site **Palm or dorsal** (Sc=40000, Lc=1200, Fc=25).

See [Running Experiments](#running-experiments) for full commands.

---

### `run_fullhand_experiment.py` — Full Hand experiment

| | |
|---|---|
| **Purpose** | Compute Q1–Q9 + unified score for full-hand captures (entire hand including fingers) |
| **When to use** | When images cover the full hand, not palm-only or dorsal-only ROI |
| **Required inputs** | Images under `data/full_hand/…`; PC maps under `debug_openvein_features/full_hand/…` |
| **Output** | `results/full_hand/` — same Excel/CSV/log schema as other modalities |
| **Capture site** | Full hand (Sc=300000, Lc=9000, Fc=50) via `vascular_quality/full_hand/config.py` |

```powershell
python run_fullhand_experiment.py `
  --qualities high_quality low_quality `
  --output results/full_hand `
  --save-excel
```

---

### `run_all_q1_q9.py` — multi-extractor research comparison

| | |
|---|---|
| **Purpose** | Evaluate **all six** OpenVein extractors (RLT, MC, WLD, PC, GF, EMC); compare unified scores |
| **When to use** | Research / extractor selection — not the production PC-only experiment |
| **Output** | `results/{EXTRACTOR}/csv/per_image_results.csv`, `results/final_summary/extractor_comparison.csv` |

```powershell
python run_all_q1_q9.py --dry-run
python run_all_q1_q9.py
python run_all_q1_q9.py --extractor PC --dataset PLUS --quality all --save-debug-images
```

---

### `run_all_q1_q7.py` — Q1–Q7 only (development)

Computes Q1–Q7 on raw images **without** OpenVein vein maps (no Q8, Q9, or unified score).

```powershell
python run_all_q1_q7.py --dataset PLUS --quality all
python run_all_q1_q7.py --input "data/finger_vein/PLUS/high_quality/your_image.png"
```

---

### `vascular_quality.finger_vein.runner` — interactive console table

Runs Q1–Q9 on one dataset and prints a table (useful for quick checks; does not write Excel).

```powershell
python -m vascular_quality.finger_vein.runner --dataset PLUS --quality high_quality --extractor PC --vessel-cleanup heuristic_default
python -m vascular_quality.finger_vein.runner --dataset SCUT --quality low_quality --extractor PC --save-debug-images
```

---

## ISO compliance: Q8 / Q9 vessel skeleton

ISO/IEC 29794-9 **Clause 5.2.8** (Q8) and **5.2.9** (Q9) define these normative steps on the vessel map within foreground **R**:

| Step | Requirement |
|------|-------------|
| **b)** | Binarize the vessel map inside **R** |
| **c)** | Thin to a 1-pixel skeleton; count skeleton pixels (Q8) or endpoints / intersections (Q9) |

Thinning uses Zhang–Suen (reference [5]). The standard does **not** define how to derive the vessel map; this project uses OpenVein MATLAB exports (`uint8` 0/255).

Q8 and Q9 scores are then normalized by capture-site coefficients **Lc** and **Fc** (see [Capture-site coefficients](#capture-site-coefficients-isoiec-29794-9)). Full Hand uses larger normalizers than Palm/Dorsal ROI.

### What is **not** in ISO

`OpenVeinVesselCleanupConfig` in `vessel_utils.py` provides optional heuristics for noisy PC maps. They are **not** normative and **can change Q8/Q9**:

| Heuristic | Effect |
|-----------|--------|
| Remove small connected components | Drops speckle; lowers **n_vessels**, feature counts |
| Clip to ROI interior (`roi_margin=3`) | Mild edge cleanup inside unoccluded **R** |
| Fill small holes | Reduces loop skeletons and false intersections |
| Morphological opening (5×5) | Erodes spurs before thinning |
| Spur pruning / short-branch removal | Lowers endpoint and intersection counts |

`remove_border_touching` is **off** in `heuristic_default` because finger-vein PC maps often touch the image frame edge.

### Presets

| Preset | CLI `--vessel-cleanup` | Use when |
|--------|------------------------|----------|
| **`heuristic_default`** | `heuristic_default` | Production PC experiments (default). Document in publications. |
| **`iso_minimal`** | `iso_minimal` | Strict ISO steps b–c only: binarize within **R**, thin once. |

Programmatic control:

```python
from vessel_utils import OpenVeinVesselCleanupConfig

cleanup = OpenVeinVesselCleanupConfig.heuristic_default()
cleanup = OpenVeinVesselCleanupConfig(fill_small_holes=False, morphological_opening=False)
```

---

## Debug visualizations (optional)

Debug PNGs are **off by default** in `run_finger_vein_experiment.py`. Enable with `--save-debug-images` or use `finger_vein.runner`.

Per-image folder: `debug_outputs/finger_vein/{DATASET}/{quality}/{image_stem}/`

| File | Content |
|------|---------|
| `original.png` | Input grayscale image |
| `pc_feature_raw.png` | OpenVein PC map |
| `pc_binary_before_cleaning.png` | Binarized within **R** |
| `pc_binary_after_cleaning.png` | After heuristic cleanup |
| `pc_skeleton.png` | 1-pixel skeleton |
| `q8_skeleton_used.png` | Skeleton passed to Q8 |
| `q9_feature_points.png` | Skeleton + endpoints / intersections |
| `q1_debug.png` … `q9_debug.png` | Per-metric overlays |

```powershell
python -m vascular_quality.finger_vein.runner --dataset PLUS --quality high_quality --extractor PC --save-debug-images --vessel-cleanup heuristic_default
```

---

## Project layout (GitHub)

Tracked in Git:

- Source code, tests, scripts
- Experiment entrypoints:
  - `run_finger_vein_experiment.py`
  - `run_palm_experiment.py`
  - `run_dorsal_hand_experiment.py`
  - `run_fullhand_experiment.py`
- Modality packages:
  - `vascular_quality/finger_vein/`
  - `vascular_quality/palm/`
  - `vascular_quality/dorsal_hand/`
  - `vascular_quality/full_hand/` (config: `DEFAULT_CAPTURE_SITE = FULL_HAND`)
- Shared coefficients: `iso_constants.py` (`CAPTURE_SITE_COEFFICIENTS`)
- `data/finger_vein/**/.gitkeep` — empty dataset folders
- `debug_openvein_features/**/PC/.gitkeep` — expected PC output folders

Ignored (local only):

- Image files under `data/` and `debug_openvein_features/`
- `results/`, `debug_outputs/`, and other generated artifacts

---

## ISO metric modules

Core implementations at repo root: `q1.py` … `q9.py`, `unified_quality.py`, `iso_*.py`, `vessel_utils.py`.

Capture-site coefficients (Sc, Lc, Fc) live in `iso_constants.py` and are applied by Q1, Q8, and Q9 according to the modality’s configured `CaptureSite`. Full Hand uses different coefficients from Palm/Dorsal ROI, following the ISO/IEC 29794-9 draft.

Unified score uses an **experimental equal-weight baseline** (`αᵢ = 1/9`) until calibrated ISO coefficients are available.

---

## Running Experiments

This section is a command-level reference for reproducing experiments from a clean checkout.
It is based on the currently implemented CLIs in this repository.

### Finger Vein

Full production experiment command:

```powershell
python run_finger_vein_experiment.py --extractor PC --datasets PLUS IDIAP SCUT --qualities high_quality low_quality --output results/finger_vein/PC --save-excel --vessel-cleanup heuristic_default
```

Arguments:

| Argument | Meaning |
|---|---|
| `--extractor PC` | Use OpenVein Principal Curvature (the only supported extractor in this production runner). |
| `--datasets PLUS IDIAP SCUT` | Run every supported finger-vein dataset. |
| `--qualities high_quality low_quality` | Run both supported quality folders. |
| `--output results/finger_vein/PC` | Write CSV/Excel/log outputs under this folder. |
| `--save-excel` | Enable Excel exports (`.xlsx`). |
| `--vessel-cleanup heuristic_default` | Use the default non-normative cleanup preset for PC maps before Q8/Q9 skeletonization. |

Supported values (finger vein):

- Datasets: `PLUS`, `IDIAP`, `SCUT`
- Qualities: `high_quality`, `low_quality`
- Extractor in production runner: `PC`
- Capture site: finger second phalanx

Supported operational options:

- `--output`: yes
- `--save-excel`: yes
- Progress reporting: yes (`--progress-every`, `--quiet`, optional `tqdm`)
- Disable debug image generation for large runs: default behavior (do **not** pass `--save-debug-images`)

Outputs generated:

- `q1_q9_pc_results.csv`
- `q1_q9_pc_results.xlsx` (when `--save-excel` is set)
- `q1_q9_pc_summary.xlsx` (when `--save-excel` is set)
- `q1_q9_pc_log.txt`

### Dorsal Hand Vein

Full experiment command (auto-detect datasets under `data/dorsal_hand/`):

```powershell
python run_dorsal_hand_experiment.py --qualities high_quality low_quality --output results/dorsal_hand --save-excel --save-csv --no-debug --progress 10
```

Supported values:

- Datasets: auto-detected — Layout A folders under `data/dorsal_hand/{DATASET}/`, or flat Layout B when `data/dorsal_hand/{high_quality|low_quality}/` exists
- Qualities: any folder names passed in `--qualities` (commonly `high_quality low_quality`)
- Capture site: palm/dorsal ROI (Sc=40000, Lc=1200, Fc=25)
- Extractor: `PC` (same OpenVein map contract as production runner)

Supported operational options:

- `--output`: yes
- `--save-excel`: yes
- `--save-csv`: yes
- Progress reporting: yes (`--progress N`, elapsed, ETA, final summary)
- Disable debug image generation: yes (`--no-debug`)

### Palm Vein

Full experiment command (auto-detect datasets under `data/palm/`):

```powershell
python run_palm_experiment.py --qualities high_quality low_quality --output results/palm --save-excel --save-csv --no-debug --progress 10
```

Supported values:

- Datasets: auto-detected — Layout A folders under `data/palm/{DATASET}/`, or flat Layout B when `data/palm/{high_quality|low_quality}/` exists
- Qualities: any folder names passed in `--qualities` (commonly `high_quality low_quality`)
- Capture site: palm/dorsal ROI (Sc=40000, Lc=1200, Fc=25)
- Extractor: `PC` (same OpenVein map contract as production runner)

Supported operational options:

- `--output`: yes
- `--save-excel`: yes
- `--save-csv`: yes
- Progress reporting: yes (`--progress N`, elapsed, ETA, final summary)
- Disable debug image generation: yes (`--no-debug`)

### Full Hand

Full Hand captures the **entire hand including fingers**. Use this modality (and its coefficients) when the ROI is not limited to palm-only or dorsal-only surface.

ISO/IEC 29794-9 Full Hand normalization coefficients (from `iso_constants.py` / `vascular_quality/full_hand/config.py`):

| Coefficient | Value |
|-------------|------:|
| Sc (effective area) | 300,000 |
| Lc (total vascular length) | 9,000 |
| Fc / Pc (feature points) | 50 |

These differ from Palm/Dorsal ROI (Sc=40,000, Lc=1,200, Fc=25).

Full experiment command (auto-detect datasets under `data/full_hand/`):

```powershell
python run_fullhand_experiment.py `
  --qualities high_quality low_quality `
  --output results/full_hand `
  --save-excel
```

Equivalent one-liner:

```powershell
python run_fullhand_experiment.py --qualities high_quality low_quality --output results/full_hand --save-excel --save-csv --no-debug --progress 10
```

Supported values:

- Datasets: auto-detected — Layout A under `data/full_hand/{DATASET}/`, or flat Layout B when `data/full_hand/{high_quality|low_quality}/` exists
- Qualities: commonly `high_quality low_quality`
- Capture site: `FULL_HAND` (Sc=300000, Lc=9000, Fc=50)
- Extractor: `PC` (same OpenVein map contract)

Same CLI flags as palm/dorsal: `--output`, `--save-excel`, `--save-csv`, `--no-debug`, `--progress`.

Place Full Hand images under:

```text
data/full_hand/{DATASET}/{high_quality|low_quality}/   # Layout A
# or
data/full_hand/{high_quality|low_quality}/             # Layout B
```

PC feature maps should be generated under `debug_openvein_features/full_hand/…` before running the experiment.

### Modality Differences vs Finger Vein

1. Finger vein runner is fixed to known datasets (`PLUS IDIAP SCUT`) and includes extra options like `--extractor`, `--dry-run`, and `--vessel-cleanup`.
2. Dorsal/palm/full-hand runners auto-detect dataset names from folder structure and expose the unified CLI: `--datasets --qualities --output --save-excel --save-csv --no-debug --progress`.
3. All modalities reuse the same Q1–Q9 engine; capture-site coefficients differ: finger=`FINGER_SECOND_PHALANX`, palm/dorsal=`PALM_OR_DORSAL`, full hand=`FULL_HAND`.

### Output Folder Structure

Current implemented layout:

```text
results/
  finger_vein/
    PC/
      q1_q9_pc_results.csv
      q1_q9_pc_results.xlsx
      q1_q9_pc_summary.xlsx
      q1_q9_pc_log.txt
  dorsal_hand/
    q1_q9_pc_results.csv
    q1_q9_pc_results.xlsx
    q1_q9_pc_summary.xlsx
    q1_q9_pc_log.txt
  palm/
    q1_q9_pc_results.csv
    q1_q9_pc_results.xlsx
    q1_q9_pc_summary.xlsx
    q1_q9_pc_log.txt
  full_hand/
    q1_q9_pc_results.csv
    q1_q9_pc_results.xlsx
    q1_q9_pc_summary.xlsx
    q1_q9_pc_log.txt
```

### Excel/CSV Columns (production export)

`q1_q9_pc_results.csv` and the `per_image_results` sheet in `q1_q9_pc_results.xlsx` contain:

| Column | Meaning |
|---|---|
| `metric_modality` | Modality label (`finger_vein`, `palm`, `dorsal_hand`, or `full_hand`). |
| `dataset` | Dataset name (e.g. `PLUS`, `IDIAP`, `SCUT`, or a named hand dataset; empty for flat Layout B). |
| `quality_folder` | Quality partition (`high_quality` or `low_quality`). |
| `extractor` | OpenVein extractor tag (`PC`). |
| `image_name` | Image filename only (no absolute path). |
| `vessel_cleanup` | Vessel cleanup preset used for Q8/Q9 preprocessing. |
| `Q1` | Effective area score (normalized by capture-site **Sc**). |
| `Q2` | Offset complement / centering score. |
| `Q3` | Gray-level spread related score. |
| `Q4` | Gray-level distribution quality score. |
| `Q5` | Entropy-based information content score. |
| `Q6` | Sharpness score. |
| `Q7` | Contrast consistency score. |
| `Q8` | Total vascular length proxy (skeleton count normalized by capture-site **Lc**). |
| `Q9` | Feature points quality from endpoints/intersections (normalized by capture-site **Fc**). |
| `n_vessels` | Skeleton pixel count used for Q8 (`N_vessel`). |
| `endpoints` | Endpoint count used in Q9 (`N_end`). |
| `intersections` | Intersection count used in Q9 (`N_int`). |
| `unified_score` | Unified quality score from Q1–Q9. |

`q1_q9_pc_summary.xlsx` contains `summary_by_dataset_quality` with grouped statistics:

- `dataset`, `quality_folder`, `extractor`, `n_images`
- For each metric (`Q1`..`Q9`, `unified_score`): `_mean`, `_std`, `_min`, `_max`

### Progress Reporting During Long Runs

Progress behavior in the production runners:

- Group header at dataset/quality start
- Per-image updates with elapsed time and ETA
- Total processed count and percentage
- Final run summary with processed/failed/skipped counts
- Optional `tqdm` total bar if installed (finger vein)

Useful flags:

- Finger vein: `--progress-every N`, `--quiet`
- Palm / dorsal / full hand: `--progress N`

### Disabling Debug Images for Large Experiments

For large runs (>1000 images), debug output is already disabled by default.

- Finger vein: do not pass `--save-debug-images`
- Palm / dorsal / full hand: pass `--no-debug` (or rely on default off behavior)
- Enable debug only when troubleshooting a small subset

### Example Runtime Commands

Finger vein (full):

```powershell
python run_finger_vein_experiment.py --extractor PC --datasets PLUS IDIAP SCUT --qualities high_quality low_quality --output results/finger_vein/PC --save-excel --vessel-cleanup heuristic_default --progress-every 10
```

Finger vein (dry-run validation):

```powershell
python run_finger_vein_experiment.py --extractor PC --datasets PLUS IDIAP SCUT --qualities high_quality low_quality --output results/finger_vein/PC --save-excel --vessel-cleanup heuristic_default --dry-run
```

Dorsal hand (auto dataset detection):

```powershell
python run_dorsal_hand_experiment.py --qualities high_quality low_quality --output results/dorsal_hand --save-excel --save-csv --no-debug --progress 10
```

Palm (auto dataset detection):

```powershell
python run_palm_experiment.py --qualities high_quality low_quality --output results/palm --save-excel --save-csv --no-debug --progress 10
```

Full hand (auto dataset detection):

```powershell
python run_fullhand_experiment.py --qualities high_quality low_quality --output results/full_hand --save-excel --save-csv --no-debug --progress 10
```

### Troubleshooting

#### Missing dataset folders

If dry-run reports missing input folders, confirm your layout is:

```text
data/finger_vein/{PLUS|IDIAP|SCUT}/{high_quality|low_quality}/

Dorsal hand / palm / full hand — either layout:
  data/{modality}/{DATASET}/{high_quality|low_quality}/     (Layout A)
  data/{modality}/{high_quality|low_quality}/               (Layout B)
```

#### Unsupported dataset names

Production finger-vein runner accepts only:

- `PLUS`
- `IDIAP`
- `SCUT`

Any other name fails argument validation.

#### Missing PC feature maps

If the experiment reports missing vein maps, generate OpenVein PC outputs first.

Finger vein:

```powershell
python -m vascular_quality.openvein.pipeline --backend matlab --matlab-toolkit-root "C:/Users/user/Downloads/OpenVein-Toolkit_v1.0.2" --dataset all --quality all --extractors PC --output debug_openvein_features --clean-output --continue-on-error
```

Full hand:

```powershell
python -m vascular_quality.openvein.pipeline --modality full_hand --backend matlab --matlab-toolkit-root "C:/Users/user/Downloads/OpenVein-Toolkit_v1.0.2" --quality all --extractors PC --output debug_openvein_features --continue-on-error
```

#### MATLAB / OpenVein backend issues

- Verify MATLAB Engine import:

```powershell
python -c "import matlab.engine; print('MATLAB Engine OK')"
```

- Verify extractor availability:

```powershell
python -m vascular_quality.openvein.pipeline --list-extractors
```

- Run pipeline dry-run for diagnostics:

```powershell
python -m vascular_quality.openvein.pipeline --backend matlab --matlab-toolkit-root "C:/Users/user/Downloads/OpenVein-Toolkit_v1.0.2" --dataset all --quality all --extractors PC --dry-run
```
