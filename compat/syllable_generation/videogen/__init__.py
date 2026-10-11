"""
[DEPRECATED SHIM] syllable_generation/videogen/__init__.py
==================================================
Relocated to: studies/behavior/videogen/__init__.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.videogen is deprecated; import from studies.behavior.videogen instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.videogen")
sys.modules[__name__] = _real

def __getattr__(name):
    return getattr(_real, name)

def __dir__():
    return dir(_real)

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
