"""
[DEPRECATED SHIM] syllable_generation/analysis/deep_research_experiments/action_prediction_error/run_ppe_analysis.py
==================================================
Relocated to: studies/behavior/deep_research_experiments/action_prediction_error/run_ppe_analysis.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.analysis.deep_research_experiments.action_prediction_error.run_ppe_analysis is deprecated; import from studies.behavior.deep_research_experiments.action_prediction_error.run_ppe_analysis instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.deep_research_experiments.action_prediction_error.run_ppe_analysis")
sys.modules[__name__] = _real

def __getattr__(name):
    return getattr(_real, name)

def __dir__():
    return dir(_real)

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
