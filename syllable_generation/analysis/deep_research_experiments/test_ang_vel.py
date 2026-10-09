"""
[DEPRECATED SHIM] syllable_generation/analysis/deep_research_experiments/test_ang_vel.py
==================================================
Relocated to: studies/behavior/deep_research_experiments/test_ang_vel.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.analysis.deep_research_experiments.test_ang_vel is deprecated; import from studies.behavior.deep_research_experiments.test_ang_vel instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.deep_research_experiments.test_ang_vel")
sys.modules[__name__] = _real

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
