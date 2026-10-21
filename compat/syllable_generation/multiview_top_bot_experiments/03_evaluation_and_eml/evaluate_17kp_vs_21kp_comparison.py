"""
[DEPRECATED SHIM] syllable_generation/multiview_top_bot_experiments/03_evaluation_and_eml/evaluate_17kp_vs_21kp_comparison.py
==================================================
Relocated to: studies/behavior/multiview_experiments/03_evaluation_and_eml/evaluate_17kp_vs_21kp_comparison.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.multiview_top_bot_experiments.03_evaluation_and_eml.evaluate_17kp_vs_21kp_comparison is deprecated; import from studies.behavior.multiview_experiments.03_evaluation_and_eml.evaluate_17kp_vs_21kp_comparison instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.multiview_experiments.03_evaluation_and_eml.evaluate_17kp_vs_21kp_comparison")
sys.modules[__name__] = _real

def __getattr__(name):
    return getattr(_real, name)

def __dir__():
    return dir(_real)

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
