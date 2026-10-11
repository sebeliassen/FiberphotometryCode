"""
[DEPRECATED SHIM] syllable_generation/multiview_top_bot_experiments/02_training_runners/run_cohort_calibrated_retrain_seed101.py
==================================================
Relocated to: studies/behavior/multiview_experiments/02_training_runners/run_cohort_calibrated_retrain_seed101.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.multiview_top_bot_experiments.02_training_runners.run_cohort_calibrated_retrain_seed101 is deprecated; import from studies.behavior.multiview_experiments.02_training_runners.run_cohort_calibrated_retrain_seed101 instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.multiview_experiments.02_training_runners.run_cohort_calibrated_retrain_seed101")
sys.modules[__name__] = _real

def __getattr__(name):
    return getattr(_real, name)

def __dir__():
    return dir(_real)

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
