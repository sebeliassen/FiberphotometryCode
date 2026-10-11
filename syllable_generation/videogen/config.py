"""
[DEPRECATED SHIM] syllable_generation/videogen/config.py
==================================================
Relocated to: studies/behavior/videogen/config.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.videogen.config is deprecated; import from studies.behavior.videogen.config instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.videogen.config")
sys.modules[__name__] = _real

def __getattr__(name):
    return getattr(_real, name)

def __dir__():
    return dir(_real)

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
