"""
[DEPRECATED SHIM] syllable_generation/analysis/ulrik_kinematics/experiments/karen_phenotyping/run_karen_permutation_tests.py
==================================================
Relocated to: studies/behavior/kinematics_ulrik/experiments/karen_phenotyping/run_karen_permutation_tests.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.analysis.ulrik_kinematics.experiments.karen_phenotyping.run_karen_permutation_tests is deprecated; import from studies.kinematics_ulrik.experiments.karen_phenotyping.run_karen_permutation_tests instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.kinematics_ulrik.experiments.karen_phenotyping.run_karen_permutation_tests")
sys.modules[__name__] = _real

def __getattr__(name):
    return getattr(_real, name)

def __dir__():
    return dir(_real)

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
