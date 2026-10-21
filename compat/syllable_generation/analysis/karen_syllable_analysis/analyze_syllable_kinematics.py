"""
[DEPRECATED SHIM] syllable_generation/analysis/karen_syllable_analysis/analyze_syllable_kinematics.py
==================================================
Relocated to: studies/behavior/karen_studies/analyze_syllable_kinematics.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.analysis.karen_syllable_analysis.analyze_syllable_kinematics is deprecated; import from studies.behavior.karen_studies.analyze_syllable_kinematics instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.karen_studies.analyze_syllable_kinematics")
sys.modules[__name__] = _real

def __getattr__(name):
    return getattr(_real, name)

def __dir__():
    return dir(_real)

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
