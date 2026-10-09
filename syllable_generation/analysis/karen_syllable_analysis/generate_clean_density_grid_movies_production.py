"""
[DEPRECATED SHIM] syllable_generation/analysis/karen_syllable_analysis/generate_clean_density_grid_movies_production.py
==================================================
Relocated to: studies/behavior/karen_studies/generate_clean_density_grid_movies_production.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.analysis.karen_syllable_analysis.generate_clean_density_grid_movies_production is deprecated; import from studies.behavior.karen_studies.generate_clean_density_grid_movies_production instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.karen_studies.generate_clean_density_grid_movies_production")
sys.modules[__name__] = _real

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
