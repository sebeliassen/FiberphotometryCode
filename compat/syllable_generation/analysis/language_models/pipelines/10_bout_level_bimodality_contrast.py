"""
[DEPRECATED SHIM] syllable_generation/analysis/language_models/pipelines/10_bout_level_bimodality_contrast.py
==================================================
Relocated to: studies/behavior/language_models/pipelines/10_bout_level_bimodality_contrast.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.analysis.language_models.pipelines.10_bout_level_bimodality_contrast is deprecated; import from studies.behavior.language_models.pipelines.10_bout_level_bimodality_contrast instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.language_models.pipelines.10_bout_level_bimodality_contrast")
sys.modules[__name__] = _real

def __getattr__(name):
    return getattr(_real, name)

def __dir__():
    return dir(_real)

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
