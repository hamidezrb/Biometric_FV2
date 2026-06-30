#!/usr/bin/env python3
"""
Finger-vein ISO Q1–Q9 experiment (PC extractor / Principal Curvature only).

Examples (bash):
  python run_finger_vein_experiment.py \\
    --extractor PC \\
    --datasets PLUS IDIAP SCUT \\
    --qualities high_quality low_quality \\
    --output results/finger_vein/PC \\
    --save-excel \\
    --dry-run

PowerShell one-liner:
  python run_finger_vein_experiment.py --extractor PC --datasets PLUS IDIAP SCUT --qualities high_quality low_quality --output results/finger_vein/PC --save-excel --dry-run
"""

from vascular_quality.finger_vein.experiment import main

if __name__ == "__main__":
    raise SystemExit(main())
