"""
[DEPRECATED SHIM] syllable_generation/analysis/language_models/lib/preprocessing/rle.py
==================================================
Relocated to: studies/behavior/language_models/lib/preprocessing/rle.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.analysis.language_models.lib.preprocessing.rle is deprecated; import from studies.behavior.language_models.lib.preprocessing.rle instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.language_models.lib.preprocessing.rle")
sys.modules[__name__] = _real

def __getattr__(name):
    return getattr(_real, name)

def __dir__():
    return dir(_real)

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
