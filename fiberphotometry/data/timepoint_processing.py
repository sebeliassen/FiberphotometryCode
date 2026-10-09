"""
[DEPRECATED SHIM] fiberphotometry/data/timepoint_processing.py
==================================================
Relocated to: studies/photometry/exploratory/data/timepoint_processing.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "fiberphotometry.data.timepoint_processing is deprecated; import from studies.photometry.exploratory.data.timepoint_processing instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.photometry.exploratory.data.timepoint_processing")
sys.modules[__name__] = _real

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
