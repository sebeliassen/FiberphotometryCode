"""
[DEPRECATED SHIM] syllable_generation/loaders/jonathan/steps/__init__.py
==================================================
Relocated to: studies/behavior/loaders/jonathan/steps/__init__.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.loaders.jonathan.steps is deprecated; import from studies.behavior.loaders.jonathan.steps instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.loaders.jonathan.steps")
sys.modules[__name__] = _real

def __getattr__(name):
    return getattr(_real, name)

def __dir__():
    return dir(_real)

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
