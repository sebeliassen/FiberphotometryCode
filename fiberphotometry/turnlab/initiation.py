"""
[DEPRECATED SHIM] fiberphotometry/turnlab/initiation.py
==================================================
Relocated to: studies/photometry/turnlab/initiation.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "fiberphotometry.turnlab.initiation is deprecated; import from studies.photometry.turnlab.initiation instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.photometry.turnlab.initiation")
sys.modules[__name__] = _real

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
