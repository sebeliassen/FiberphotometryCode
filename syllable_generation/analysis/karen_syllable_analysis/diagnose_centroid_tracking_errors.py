"""
[DEPRECATED SHIM] syllable_generation/analysis/karen_syllable_analysis/diagnose_centroid_tracking_errors.py
==================================================
Relocated to: studies/behavior/karen_studies/diagnose_centroid_tracking_errors.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.analysis.karen_syllable_analysis.diagnose_centroid_tracking_errors is deprecated; import from studies.behavior.karen_studies.diagnose_centroid_tracking_errors instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.karen_studies.diagnose_centroid_tracking_errors")
sys.modules[__name__] = _real

def __getattr__(name):
    return getattr(_real, name)

def __dir__():
    return dir(_real)

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
