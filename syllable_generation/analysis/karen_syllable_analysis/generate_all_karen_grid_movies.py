"""
[DEPRECATED SHIM] syllable_generation/analysis/karen_syllable_analysis/generate_all_karen_grid_movies.py
==================================================
Relocated to: studies/behavior/karen_studies/generate_all_karen_grid_movies.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.analysis.karen_syllable_analysis.generate_all_karen_grid_movies is deprecated; import from studies.behavior.karen_studies.generate_all_karen_grid_movies instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.karen_studies.generate_all_karen_grid_movies")
sys.modules[__name__] = _real

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
