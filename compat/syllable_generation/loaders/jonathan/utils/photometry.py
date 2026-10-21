"""
[DEPRECATED SHIM] syllable_generation/loaders/jonathan/utils/photometry.py
==================================================
Relocated to: studies/session_building/jonathan/utils/photometry.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.loaders.jonathan.utils.photometry is deprecated; import from studies.session_building.jonathan.utils.photometry instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.session_building.jonathan.utils.photometry")
sys.modules[__name__] = _real

def __getattr__(name):
    return getattr(_real, name)

def __dir__():
    return dir(_real)

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
