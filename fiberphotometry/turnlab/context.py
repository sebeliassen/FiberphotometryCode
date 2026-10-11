"""
[DEPRECATED SHIM] fiberphotometry/turnlab/context.py
==================================================
Relocated to: studies/photometry/turnlab/context.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "fiberphotometry.turnlab.context is deprecated; import from studies.photometry.turnlab.context instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.photometry.turnlab.context")
sys.modules[__name__] = _real

def __getattr__(name):
    return getattr(_real, name)

def __dir__():
    return dir(_real)

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
