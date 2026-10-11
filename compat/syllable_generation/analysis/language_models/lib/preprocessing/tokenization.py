"""
[DEPRECATED SHIM] syllable_generation/analysis/language_models/lib/preprocessing/tokenization.py
==================================================
Relocated to: studies/behavior/language_models/lib/preprocessing/tokenization.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.analysis.language_models.lib.preprocessing.tokenization is deprecated; import from studies.behavior.language_models.lib.preprocessing.tokenization instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.language_models.lib.preprocessing.tokenization")
sys.modules[__name__] = _real

def __getattr__(name):
    return getattr(_real, name)

def __dir__():
    return dir(_real)

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
