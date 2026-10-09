"""
[DEPRECATED SHIM] syllable_generation/analysis/ulrik_kinematics/lib/aggregation.py
==================================================
Relocated to: studies/behavior/kinematics_ulrik/lib/aggregation.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.analysis.ulrik_kinematics.lib.aggregation is deprecated; import from studies.behavior.kinematics_ulrik.lib.aggregation instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.kinematics_ulrik.lib.aggregation")
sys.modules[__name__] = _real

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
