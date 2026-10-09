"""
[DEPRECATED SHIM] syllable_generation/analysis/karen_syllable_analysis/generate_all_density_clean_grid_movies_vfi.py
==================================================
Relocated to: studies/behavior/karen_studies/generate_all_density_clean_grid_movies_vfi.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.analysis.karen_syllable_analysis.generate_all_density_clean_grid_movies_vfi is deprecated; import from studies.behavior.karen_studies.generate_all_density_clean_grid_movies_vfi instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.karen_studies.generate_all_density_clean_grid_movies_vfi")
sys.modules[__name__] = _real

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
