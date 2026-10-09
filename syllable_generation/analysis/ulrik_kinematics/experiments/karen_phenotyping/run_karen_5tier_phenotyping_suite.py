"""
[DEPRECATED SHIM] syllable_generation/analysis/ulrik_kinematics/experiments/karen_phenotyping/run_karen_5tier_phenotyping_suite.py
==================================================
Relocated to: studies/behavior/kinematics_ulrik/experiments/karen_phenotyping/run_karen_5tier_phenotyping_suite.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.analysis.ulrik_kinematics.experiments.karen_phenotyping.run_karen_5tier_phenotyping_suite is deprecated; import from studies.behavior.kinematics_ulrik.experiments.karen_phenotyping.run_karen_5tier_phenotyping_suite instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.kinematics_ulrik.experiments.karen_phenotyping.run_karen_5tier_phenotyping_suite")
sys.modules[__name__] = _real

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
