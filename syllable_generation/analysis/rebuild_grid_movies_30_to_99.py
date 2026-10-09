"""
[DEPRECATED SHIM] syllable_generation/analysis/rebuild_grid_movies_30_to_99.py
==================================================
Relocated to: studies/behavior/misc/syllable_generation/analysis/rebuild_grid_movies_30_to_99.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.analysis.rebuild_grid_movies_30_to_99 is deprecated; import from studies.behavior.misc.syllable_generation.analysis.rebuild_grid_movies_30_to_99 instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.misc.syllable_generation.analysis.rebuild_grid_movies_30_to_99")
sys.modules[__name__] = _real

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
