"""
[DEPRECATED SHIM] syllable_generation/analysis/shmoseq/test_model_fit.py
==================================================
Relocated to: studies/behavior/shmoseq/test_model_fit.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.analysis.shmoseq.test_model_fit is deprecated; import from studies.behavior.shmoseq.test_model_fit instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.shmoseq.test_model_fit")
sys.modules[__name__] = _real

def __getattr__(name):
    return getattr(_real, name)

def __dir__():
    return dir(_real)

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
