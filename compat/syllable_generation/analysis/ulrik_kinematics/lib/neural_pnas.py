"""
[DEPRECATED SHIM] syllable_generation/analysis/ulrik_kinematics/lib/neural_pnas.py
==================================================
Relocated to: studies/behavior/kinematics_ulrik/lib/neural_pnas.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.analysis.ulrik_kinematics.lib.neural_pnas is deprecated; import from studies.kinematics_ulrik.lib.neural_pnas instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.kinematics_ulrik.lib.neural_pnas")
sys.modules[__name__] = _real

def __getattr__(name):
    return getattr(_real, name)

def __dir__():
    return dir(_real)

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
