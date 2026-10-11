"""
[DEPRECATED SHIM] fiberphotometry/data/mouse.py
==================================================
Relocated to: studies/photometry/exploratory/data/mouse.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "fiberphotometry.data.mouse is deprecated; import from studies.photometry.exploratory.data.mouse instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.photometry.exploratory.data.mouse")
sys.modules[__name__] = _real

def __getattr__(name):
    return getattr(_real, name)

def __dir__():
    return dir(_real)

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
