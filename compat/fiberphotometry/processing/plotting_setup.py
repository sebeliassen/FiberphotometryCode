"""
[DEPRECATED SHIM] fiberphotometry/processing/plotting_setup.py
==================================================
Relocated to: studies/photometry/exploratory/processing/plotting_setup.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "fiberphotometry.processing.plotting_setup is deprecated; import from studies.photometry.exploratory.processing.plotting_setup instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.photometry.exploratory.processing.plotting_setup")
sys.modules[__name__] = _real

def __getattr__(name):
    return getattr(_real, name)

def __dir__():
    return dir(_real)

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
