"""
[DEPRECATED SHIM] fiberphotometry/analysis/plotting.py
==================================================
Relocated to: studies/photometry/exploratory/analysis/plotting.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "fiberphotometry.analysis.plotting is deprecated; import from studies.photometry.exploratory.analysis.plotting instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.photometry.exploratory.analysis.plotting")
sys.modules[__name__] = _real

def __getattr__(name):
    return getattr(_real, name)

def __dir__():
    return dir(_real)

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
