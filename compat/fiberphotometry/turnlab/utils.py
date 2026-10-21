"""
[DEPRECATED SHIM] fiberphotometry/turnlab/utils.py
==================================================
Relocated to: studies/photometry/turnlab/utils.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "fiberphotometry.turnlab.utils is deprecated; import from studies.photometry.turnlab.utils instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.photometry.turnlab.utils")
sys.modules[__name__] = _real

def __getattr__(name):
    return getattr(_real, name)

def __dir__():
    return dir(_real)

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
