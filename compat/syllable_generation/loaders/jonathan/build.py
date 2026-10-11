"""
[DEPRECATED SHIM] syllable_generation/loaders/jonathan/build.py
==================================================
Relocated to: studies/behavior/loaders/jonathan/build.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.loaders.jonathan.build is deprecated; import from studies.behavior.loaders.jonathan.build instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.loaders.jonathan.build")
sys.modules[__name__] = _real

def __getattr__(name):
    return getattr(_real, name)

def __dir__():
    return dir(_real)

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
