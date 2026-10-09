"""
[DEPRECATED SHIM] syllable_generation/multiview_top_bot_experiments/01_data_and_fusion/audit_cohort_basler_calibrated.py
==================================================
Relocated to: studies/behavior/multiview_experiments/01_data_and_fusion/audit_cohort_basler_calibrated.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.multiview_top_bot_experiments.01_data_and_fusion.audit_cohort_basler_calibrated is deprecated; import from studies.behavior.multiview_experiments.01_data_and_fusion.audit_cohort_basler_calibrated instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.multiview_experiments.01_data_and_fusion.audit_cohort_basler_calibrated")
sys.modules[__name__] = _real

def __getattr__(name):
    return getattr(_real, name)

def __dir__():
    return dir(_real)

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
