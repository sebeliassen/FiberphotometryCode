"""
[DEPRECATED SHIM] fiberphotometry/analysis/timepoint_analysis_leonie.py
==================================================
Relocated to: studies/photometry/exploratory/analysis/timepoint_analysis_leonie.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "fiberphotometry.analysis.timepoint_analysis_leonie is deprecated; import from studies.photometry.exploratory.analysis.timepoint_analysis_leonie instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.photometry.exploratory.analysis.timepoint_analysis_leonie")
sys.modules[__name__] = _real

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
