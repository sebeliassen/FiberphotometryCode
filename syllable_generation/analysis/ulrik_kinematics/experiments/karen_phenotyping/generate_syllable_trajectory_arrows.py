"""
[DEPRECATED SHIM] syllable_generation/analysis/ulrik_kinematics/experiments/karen_phenotyping/generate_syllable_trajectory_arrows.py
==================================================
Relocated to: studies/behavior/kinematics_ulrik/experiments/karen_phenotyping/generate_syllable_trajectory_arrows.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.analysis.ulrik_kinematics.experiments.karen_phenotyping.generate_syllable_trajectory_arrows is deprecated; import from studies.behavior.kinematics_ulrik.experiments.karen_phenotyping.generate_syllable_trajectory_arrows instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.kinematics_ulrik.experiments.karen_phenotyping.generate_syllable_trajectory_arrows")
sys.modules[__name__] = _real

def __getattr__(name):
    return getattr(_real, name)

def __dir__():
    return dir(_real)

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
