"""
[DEPRECATED SHIM] syllable_generation/analysis/karen_syllable_analysis/transparent_animal_permutation_analysis.py
==================================================
Relocated to: studies/behavior/karen_studies/transparent_animal_permutation_analysis.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.analysis.karen_syllable_analysis.transparent_animal_permutation_analysis is deprecated; import from studies.behavior.karen_studies.transparent_animal_permutation_analysis instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.karen_studies.transparent_animal_permutation_analysis")
sys.modules[__name__] = _real

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
