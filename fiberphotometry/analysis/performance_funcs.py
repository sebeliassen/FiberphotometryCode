"""
[DEPRECATED SHIM] fiberphotometry/analysis/performance_funcs.py
==================================================
Relocated to: studies/photometry/exploratory/analysis/performance_funcs.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "fiberphotometry.analysis.performance_funcs is deprecated; import from studies.photometry.exploratory.analysis.performance_funcs instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.photometry.exploratory.analysis.performance_funcs")
sys.modules[__name__] = _real

def __getattr__(name):
    return getattr(_real, name)

def __dir__():
    return dir(_real)

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
