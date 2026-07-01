# ISO/IEC 29794-9 Finger-Vein Quality Assessment

Python implementation of ISO/IEC 29794-9 quality components (Q1–Q9) for finger vascular biometrics, with OpenVein feature extraction via the MATLAB Engine.

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
data/finger_vein/{DATASET}/{quality}/          ← input images (not in Git)
        ↓
vascular_quality.openvein.pipeline             ← OpenVein PC extraction (MATLAB)
        ↓
debug_openvein_features/{DATASET}/{quality}/PC/  ← vein maps (not in Git)
        ↓
run_finger_vein_experiment.py                  ← ISO Q1–Q9 + unified score
        ↓
results/finger_vein/PC/                        ← Excel / CSV / log
```

### Recommended workflow (production)

1. **Extract PC features** (MATLAB) for all datasets and qualities.
2. **Dry-run** the quality experiment — verify image and PC map counts match.
3. **Smoke test** with `--limit 30` on a subset.
4. **Full experiment** with `--progress-every 10` (debug images off by default).

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
```

Supported formats: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff`

**Git:** folder structure is tracked via `.gitkeep` files; **image files are ignored** and must be added locally on each machine.

Optional one-time setup (creates folders + migrates legacy layouts):

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

### Extract OpenVein features (MATLAB backend)

Single dataset / quality:

```powershell
python -m vascular_quality.openvein.pipeline --backend matlab --matlab-toolkit-root "C:/Users/user/Downloads/OpenVein-Toolkit_v1.0.2" --dataset PLUS --quality high_quality --extractors PC --output debug_openvein_features --continue-on-error
```

All datasets and qualities (`--dataset all` → PLUS, IDIAP, SCUT; `--quality all` → high_quality, low_quality):

```powershell
python -m vascular_quality.openvein.pipeline --backend matlab --matlab-toolkit-root "C:/Users/user/Downloads/OpenVein-Toolkit_v1.0.2" --dataset all --quality all --extractors PC --output debug_openvein_features --clean-output --continue-on-error
```

Dry-run (validate paths and counts, no extraction):

```powershell
python -m vascular_quality.openvein.pipeline --backend matlab --matlab-toolkit-root "C:/Users/user/Downloads/OpenVein-Toolkit_v1.0.2" --dataset all --quality all --extractors PC --dry-run
```

Output layout:

```text
debug_openvein_features/
  {PLUS|IDIAP|SCUT}/
    {high_quality|low_quality}/
      PC/
        {stem}.png
```

| Flag | Purpose |
|------|---------|
| `--dataset all` | Run PLUS, IDIAP, and SCUT |
| `--quality all` | Run high_quality and low_quality |
| `--continue-on-error` | Log a failed job and continue |
| `--clean-output` | Remove stale vein maps before extraction |
| `--limit N` | Process only the first N images (smoke test) |
| `--backend python` | Approximate fallback only (not for final experiments) |

---

## Main experiment scripts

### `run_finger_vein_experiment.py` — production PC experiment

| | |
|---|---|
| **Purpose** | Compute Q1–Q9 + unified score using **PC (Principal Curvature)** OpenVein maps; export Excel/CSV for statistical analysis |
| **When to use** | Final experiments and thesis analysis (single extractor: PC) |
| **Required inputs** | Images in `data/finger_vein/{DATASET}/{quality}/`; PC maps in `debug_openvein_features/{DATASET}/{quality}/PC/{stem}.png` |
| **Output** | `q1_q9_pc_results.xlsx`, `q1_q9_pc_results.csv`, `q1_q9_pc_summary.xlsx`, `q1_q9_pc_log.txt` |

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
- `data/finger_vein/**/.gitkeep` — empty dataset folders
- `debug_openvein_features/**/PC/.gitkeep` — expected PC output folders

Ignored (local only):

- Image files under `data/` and `debug_openvein_features/`
- `results/`, `debug_outputs/`, and other generated artifacts

---

## ISO metric modules

Core implementations at repo root: `q1.py` … `q9.py`, `unified_quality.py`, `iso_*.py`, `vessel_utils.py`.

Unified score uses an **experimental equal-weight baseline** (`αᵢ = 1/9`) until calibrated ISO coefficients are available.
