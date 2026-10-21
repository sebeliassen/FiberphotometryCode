"""
[DEPRECATED SHIM] fiberphotometry/analysis/timepoint_analysis.py
==================================================
Relocated to: studies/photometry/exploratory/analysis/timepoint_analysis.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "fiberphotometry.analysis.timepoint_analysis is deprecated; import from studies.photometry.exploratory.analysis.timepoint_analysis instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.photometry.exploratory.analysis.timepoint_analysis")
sys.modules[__name__] = _real

def __getattr__(name):
    return getattr(_real, name)

def __dir__():
    return dir(_real)

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
