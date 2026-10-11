"""
[DEPRECATED SHIM] syllable_generation/analysis/language_models/nmf_review_package/code/build_run12_clean_bout_cache.py
==================================================
Relocated to: studies/behavior/language_models/nmf_review_package/code/build_run12_clean_bout_cache.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.analysis.language_models.nmf_review_package.code.build_run12_clean_bout_cache is deprecated; import from studies.behavior.language_models.nmf_review_package.code.build_run12_clean_bout_cache instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.language_models.nmf_review_package.code.build_run12_clean_bout_cache")
sys.modules[__name__] = _real

def __getattr__(name):
    return getattr(_real, name)

def __dir__():
    return dir(_real)

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
