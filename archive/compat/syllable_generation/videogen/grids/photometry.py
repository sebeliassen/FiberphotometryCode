"""
[DEPRECATED SHIM] syllable_generation/videogen/grids/photometry.py
==================================================
Relocated to: studies/behavior/videogen/grids/photometry.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.videogen.grids.photometry is deprecated; import from studies.behavior.videogen.grids.photometry instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.videogen.grids.photometry")
sys.modules[__name__] = _real

def __getattr__(name):
    return getattr(_real, name)

def __dir__():
    return dir(_real)

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
