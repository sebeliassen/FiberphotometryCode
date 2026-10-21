"""
[DEPRECATED SHIM] syllable_generation/metadata/karen_static_21kp_v2_labels.py
==================================================
Relocated to: studies/behavior/karen_studies/karen_static_21kp_v2_labels.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.metadata.karen_static_21kp_v2_labels is deprecated; import from studies.behavior.karen_studies.karen_static_21kp_v2_labels instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.karen_studies.karen_static_21kp_v2_labels")
sys.modules[__name__] = _real

def __getattr__(name):
    return getattr(_real, name)

def __dir__():
    return dir(_real)

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
