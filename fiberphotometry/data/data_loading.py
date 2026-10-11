"""
[DEPRECATED SHIM] fiberphotometry/data/data_loading.py
==================================================
Relocated to: studies/photometry/exploratory/data/data_loading.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "fiberphotometry.data.data_loading is deprecated; import from studies.photometry.exploratory.data.data_loading instead",
    DeprecationWarning,
    stacklevel=2,
)

from studies.photometry.exploratory.data.data_loading import Session, DataContainer

_real = importlib.import_module("studies.photometry.exploratory.data.data_loading")
sys.modules[__name__] = _real

def __getattr__(name):
    return getattr(_real, name)

def __dir__():
    return dir(_real)

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
