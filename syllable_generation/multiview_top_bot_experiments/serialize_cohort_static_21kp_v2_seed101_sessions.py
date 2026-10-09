"""
[DEPRECATED SHIM] syllable_generation/multiview_top_bot_experiments/serialize_cohort_static_21kp_v2_seed101_sessions.py
==================================================
Relocated to: studies/behavior/multiview_experiments/serialize_cohort_static_21kp_v2_seed101_sessions.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.multiview_top_bot_experiments.serialize_cohort_static_21kp_v2_seed101_sessions is deprecated; import from studies.behavior.multiview_experiments.serialize_cohort_static_21kp_v2_seed101_sessions instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.multiview_experiments.serialize_cohort_static_21kp_v2_seed101_sessions")
sys.modules[__name__] = _real

def __getattr__(name):
    return getattr(_real, name)

def __dir__():
    return dir(_real)

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
