"""
[DEPRECATED SHIM] fiberphotometry/turnlab/linking.py
==================================================
Relocated to: studies/photometry/turnlab/linking.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "fiberphotometry.turnlab.linking is deprecated; import from studies.photometry.turnlab.linking instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.photometry.turnlab.linking")
sys.modules[__name__] = _real

def __getattr__(name):
    return getattr(_real, name)

def __dir__():
    return dir(_real)

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
