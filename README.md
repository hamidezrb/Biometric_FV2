# ISO/IEC 29794-9 Finger-Vein Quality Assessment

Python implementation of ISO/IEC 29794-9 quality components (Q1–Q9) for finger vascular biometrics, with OpenVein feature extraction via the MATLAB Engine.

## Install

```powershell
pip install numpy opencv-python scikit-image pandas openpyxl
```

## Pipeline overview

```text
data/finger_vein/{DATASET}/{quality}/     ← input images (not in Git)
        ↓
vascular_quality.openvein.pipeline        ← OpenVein feature extraction (MATLAB)
        ↓
debug_openvein_features/.../{EXTRACTOR}/  ← vein maps (not in Git)
        ↓
run_finger_vein_experiment.py             ← ISO Q1–Q9 + unified score (PC)
        ↓
results/finger_vein/PC/                   ← Excel/CSV for analysis
```

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

OpenVein’s original algorithms (RLT, MC, WLD, **PC**, GF, EMC) ship as a MATLAB toolkit. The **MATLAB backend** (`--backend matlab`) runs those algorithms exactly through `matlab.engine` — required for Principal Curvature (PC) and for publication-grade feature maps.

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

```powershell
python -m vascular_quality.openvein.pipeline --backend matlab --matlab-toolkit-root "C:/Users/user/Downloads/OpenVein-Toolkit_v1.0.2" --dataset PLUS --quality high_quality --extractors PC --output debug_openvein_features --continue-on-error
```

All datasets, all qualities, PC only:

```powershell
python -m vascular_quality.openvein.pipeline --backend matlab --matlab-toolkit-root "C:/Users/user/Downloads/OpenVein-Toolkit_v1.0.2" --dataset all --quality all --extractors PC --clean-output
```

Output layout:

```text
debug_openvein_features/
  {PLUS|IDIAP|SCUT}/
    {high_quality|low_quality}/
      PC/
        {stem}.png
```

Dry-run (no extraction):

```powershell
python -m vascular_quality.openvein.pipeline --backend matlab --matlab-toolkit-root "C:/Users/user/Downloads/OpenVein-Toolkit_v1.0.2" --dataset PLUS --quality high_quality --extractors PC --dry-run
```

| Flag | Purpose |
|------|---------|
| `--continue-on-error` | Log a failed job and continue |
| `--clean-output` | Remove stale vein maps before extraction |
| `--backend python` | Approximate fallback only (not for final experiments) |

---

## Main experiment scripts

### `run_finger_vein_experiment.py` — production PC experiment

| | |
|---|---|
| **Purpose** | Compute Q1–Q9 + unified score using **PC (Principal Curvature)** OpenVein maps; export Excel/CSV for statistical analysis |
| **When to use** | Final experiments and thesis analysis (single extractor: PC) |
| **Required inputs** | Images in `data/finger_vein/{DATASET}/{quality}/`; PC maps in `debug_openvein_features/{DATASET}/{quality}/PC/{stem}.png` |
| **Output** | `results/finger_vein/PC/q1_q9_pc_results.xlsx`, `.csv`, `q1_q9_pc_summary.xlsx`, `q1_q9_pc_log.txt` |

Dry-run:

```powershell
python run_finger_vein_experiment.py --extractor PC --datasets PLUS IDIAP SCUT --qualities high_quality low_quality --output results/finger_vein/PC --save-excel --dry-run
```

Run experiment:

```powershell
python run_finger_vein_experiment.py --extractor PC --datasets PLUS IDIAP SCUT --qualities high_quality low_quality --output results/finger_vein/PC --save-excel
```

Export columns: `metric_modality`, `dataset`, `quality_folder`, `extractor`, `image_name`, `vessel_cleanup`, `Q1`–`Q9`, `unified_score`

Vessel cleanup preset (logged in CSV/Excel and experiment log):

```powershell
python run_finger_vein_experiment.py --extractor PC --vessel-cleanup iso_minimal ...
python run_finger_vein_experiment.py --extractor PC --vessel-cleanup heuristic_default ...
```

---

### `run_all_q1_q9.py` — multi-extractor research comparison

| | |
|---|---|
| **Purpose** | Evaluate **all six** OpenVein extractors (RLT, MC, WLD, PC, GF, EMC); compare unified scores across extractors |
| **When to use** | Research / extractor selection — not the production PC-only experiment |
| **Required inputs** | Images in `data/finger_vein/`; vein maps under `debug_openvein_features/{DATASET}/{quality}/{EXTRACTOR}/` for each extractor you run |
| **Output** | `results/{EXTRACTOR}/csv/per_image_results.csv`, `results/final_summary/extractor_comparison.csv`, etc. |

Dry-run:

```powershell
python run_all_q1_q9.py --dry-run
```

Run all extractors:

```powershell
python run_all_q1_q9.py
```

Single extractor:

```powershell
python run_all_q1_q9.py --extractor PC --dataset PLUS --quality all
```

---

### `run_all_q1_q7.py` — Q1–Q7 only (development)

| | |
|---|---|
| **Purpose** | Compute Q1–Q7 on raw images **without** OpenVein vein maps (no Q8, Q9, or unified score) |
| **When to use** | Quick checks of occlusion, contrast, and related metrics during development |
| **Required inputs** | Images in `data/finger_vein/{DATASET}/{quality}/` or `--input` path |
| **Output** | Console table; debug PNGs in `debug_outputs_q1_q7/` (gray, mask, overlay) |

Default (PLUS, all qualities):

```powershell
python run_all_q1_q7.py --dataset PLUS --quality all
```

Single image:

```powershell
python run_all_q1_q7.py --input "data/finger_vein/PLUS/high_quality/your_image.png"
```

Custom output folder:

```powershell
python run_all_q1_q7.py --dataset IDIAP --quality high_quality --out debug_outputs_q1_q7
```

---

## ISO compliance: Q8 / Q9 vessel skeleton

ISO/IEC 29794-9 **Clause 5.2.8** (Q8) and **5.2.9** (Q9) define these normative preprocessing steps on the vessel map within foreground **R**:

| Step | Requirement |
|------|-------------|
| **b)** | Binarize the vessel map inside **R** |
| **c)** | Thin to a 1-pixel skeleton; count skeleton pixels (Q8) or endpoints / intersections (Q9) |

Thinning uses Zhang–Suen (reference [5] in the standard). The standard does **not** define how to derive the vessel map from the vascular image; this project supplies OpenVein MATLAB exports (`uint8` 0/255).

### What is **not** in ISO

The following steps in `OpenVeinVesselCleanupConfig` (`vessel_utils.py`) are **heuristic post-processing** for noisy OpenVein PC maps. They are **not** normative and **can change Q8/Q9**:

| Heuristic | Effect on metrics |
|-----------|-------------------|
| Remove small connected components | Drops speckle; lowers **N_vessel**, **N_fp** |
| Remove image-border components | Drops edge artifacts (**off** in `heuristic_default`; finger PC maps touch frame border) |
| Clip to ROI interior (`roi_margin`) | Mild edge cleanup inside unoccluded **R** (default: 3 px) |
| Clip to ROI interior | Removes veins near **R** boundary |
| **Fill small holes** | Merges enclosed gaps; reduces loop skeletons and false intersections (**Q9**) |
| **Morphological opening** | Erodes thin spurs before thinning; lowers **N_vessel**, **N_fp** |
| **Spur pruning** (post-thin) | Removes short dead-end branches; lowers **N_fp** |
| Remove short skeleton branches | Drops fragmented segments; lowers **N_vessel**, **N_fp** |

### Presets (reproducible experiments)

| Preset | CLI `--vessel-cleanup` | Use when |
|--------|------------------------|----------|
| **`iso_minimal`** | `iso_minimal` | Strict ISO steps b–c: binarize OpenVein map within unoccluded **R**, thin once. No hole fill, opening, or spur pruning. |
| **`heuristic_default`** | `heuristic_default` (default) | Noisy OpenVein PC maps; cleaner skeletons for analysis. **Document this choice** in publications — it is not ISO-defined. |

Programmatic per-flag control:

```python
from vessel_utils import OpenVeinVesselCleanupConfig

cleanup = OpenVeinVesselCleanupConfig(
    fill_small_holes=False,
    morphological_opening=False,
    prune_spurs=True,
    prune_iters=30,
)
```

Unit tests (`test_q8_iso_page.py`, `test_q9_iso_page.py`) use the ISO-minimal path on synthetic maps via `vein_map_source="iso"`.

---

## Debug visualizations (optional)

ISO metric debug PNGs are **off by default** in Q1–Q9 pipelines. To generate overlays and skeletons:

```powershell
python -m vascular_quality.finger_vein.runner --dataset PLUS --quality high_quality --extractor PC --save-debug-images --vessel-cleanup heuristic_default
```

Writes to `debug_outputs/finger_vein/{DATASET}/{quality}/{image_stem}/` (includes PC skeleton stages when debug is enabled).

---

## Project layout (GitHub)

Tracked in Git:

- Source code, tests, scripts
- `data/finger_vein/**/.gitkeep` — empty dataset folders
- `debug_openvein_features/**/PC/.gitkeep` — expected PC output folders

Ignored (local only):

- All image files under `data/` and `debug_openvein_features/`
- `results/`, `debug_outputs/`, and other generated artifacts

---

## ISO metric modules

Core implementations at repo root: `q1.py` … `q9.py`, `unified_quality.py`, `iso_*.py`, `vessel_utils.py`.

Unified score uses an **experimental equal-weight baseline** (`αᵢ = 1/9`) until calibrated ISO coefficients are available.
