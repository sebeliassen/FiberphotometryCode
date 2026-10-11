"""
[DEPRECATED SHIM] syllable_generation/analysis/ulrik_kinematics/lib/clustering.py
==================================================
Relocated to: studies/behavior/kinematics_ulrik/lib/clustering.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.analysis.ulrik_kinematics.lib.clustering is deprecated; import from studies.behavior.kinematics_ulrik.lib.clustering instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.kinematics_ulrik.lib.clustering")
sys.modules[__name__] = _real

def __getattr__(name):
    return getattr(_real, name)

def __dir__():
    return dir(_real)

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
