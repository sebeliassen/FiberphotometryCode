"""
[DEPRECATED SHIM] fiberphotometry/turnlab/sanity.py
==================================================
Relocated to: studies/photometry/turnlab/sanity.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "fiberphotometry.turnlab.sanity is deprecated; import from studies.photometry.turnlab.sanity instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.photometry.turnlab.sanity")
sys.modules[__name__] = _real

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
