"""
[DEPRECATED SHIM] syllable_generation/analysis/ulrik_kinematics/experiments/karen_phenotyping/generate_all_metric_timecourses.py
==================================================
Relocated to: studies/behavior/kinematics_ulrik/experiments/karen_phenotyping/generate_all_metric_timecourses.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.analysis.ulrik_kinematics.experiments.karen_phenotyping.generate_all_metric_timecourses is deprecated; import from studies.behavior.kinematics_ulrik.experiments.karen_phenotyping.generate_all_metric_timecourses instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.kinematics_ulrik.experiments.karen_phenotyping.generate_all_metric_timecourses")
sys.modules[__name__] = _real

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
