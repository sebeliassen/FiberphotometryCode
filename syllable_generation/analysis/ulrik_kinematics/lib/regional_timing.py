"""
[DEPRECATED SHIM] syllable_generation/analysis/ulrik_kinematics/lib/regional_timing.py
==================================================
Relocated to: studies/behavior/kinematics_ulrik/lib/regional_timing.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.analysis.ulrik_kinematics.lib.regional_timing is deprecated; import from studies.behavior.kinematics_ulrik.lib.regional_timing instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.kinematics_ulrik.lib.regional_timing")
sys.modules[__name__] = _real

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
