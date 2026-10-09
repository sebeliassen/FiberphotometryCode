"""
[DEPRECATED SHIM] fiberphotometry/notebooks/sebastian/tsnepy.py
==================================================
Relocated to: studies/photometry/exploratory/notebooks/sebastian/tsnepy.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "fiberphotometry.notebooks.sebastian.tsnepy is deprecated; import from studies.photometry.exploratory.notebooks.sebastian.tsnepy instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.photometry.exploratory.notebooks.sebastian.tsnepy")
sys.modules[__name__] = _real

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
