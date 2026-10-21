"""
[DEPRECATED SHIM] fiberphotometry/analysis/response_metrics.py
==================================================
Relocated to: studies/photometry/exploratory/analysis/response_metrics.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "fiberphotometry.analysis.response_metrics is deprecated; import from studies.photometry.exploratory.analysis.response_metrics instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.photometry.exploratory.analysis.response_metrics")
sys.modules[__name__] = _real

def __getattr__(name):
    return getattr(_real, name)

def __dir__():
    return dir(_real)

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
