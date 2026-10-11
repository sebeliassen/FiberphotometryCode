"""
[DEPRECATED SHIM] fiberphotometry/processing/session_sampling.py
==================================================
Relocated to: studies/photometry/exploratory/processing/session_sampling.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "fiberphotometry.processing.session_sampling is deprecated; import from studies.photometry.exploratory.processing.session_sampling instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.photometry.exploratory.processing.session_sampling")
sys.modules[__name__] = _real

def __getattr__(name):
    return getattr(_real, name)

def __dir__():
    return dir(_real)

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
