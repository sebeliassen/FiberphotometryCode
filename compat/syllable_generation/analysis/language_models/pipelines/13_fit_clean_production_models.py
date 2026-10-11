"""
[DEPRECATED SHIM] syllable_generation/analysis/language_models/pipelines/13_fit_clean_production_models.py
==================================================
Relocated to: studies/behavior/language_models/pipelines/13_fit_clean_production_models.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.analysis.language_models.pipelines.13_fit_clean_production_models is deprecated; import from studies.behavior.language_models.pipelines.13_fit_clean_production_models instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.language_models.pipelines.13_fit_clean_production_models")
sys.modules[__name__] = _real

def __getattr__(name):
    return getattr(_real, name)

def __dir__():
    return dir(_real)

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
