"""
[DEPRECATED SHIM] fiberphotometry/processing/signal_info_setup.py
==================================================
Relocated to: studies/photometry/exploratory/processing/signal_info_setup.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "fiberphotometry.processing.signal_info_setup is deprecated; import from studies.photometry.exploratory.processing.signal_info_setup instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.photometry.exploratory.processing.signal_info_setup")
sys.modules[__name__] = _real

def __getattr__(name):
    return getattr(_real, name)

def __dir__():
    return dir(_real)

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
