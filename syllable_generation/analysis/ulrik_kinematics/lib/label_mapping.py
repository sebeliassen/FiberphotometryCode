"""
[DEPRECATED SHIM] syllable_generation/analysis/ulrik_kinematics/lib/label_mapping.py
==================================================
Relocated to: studies/behavior/kinematics_ulrik/lib/label_mapping.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.analysis.ulrik_kinematics.lib.label_mapping is deprecated; import from studies.behavior.kinematics_ulrik.lib.label_mapping instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.kinematics_ulrik.lib.label_mapping")
sys.modules[__name__] = _real

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
