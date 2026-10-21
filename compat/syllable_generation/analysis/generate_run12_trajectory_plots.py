"""
[DEPRECATED SHIM] syllable_generation/analysis/generate_run12_trajectory_plots.py
==================================================
Relocated to: studies/behavior/misc/syllable_generation/analysis/generate_run12_trajectory_plots.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.analysis.generate_run12_trajectory_plots is deprecated; import from studies.behavior.misc.syllable_generation.analysis.generate_run12_trajectory_plots instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.misc.syllable_generation.analysis.generate_run12_trajectory_plots")
sys.modules[__name__] = _real

def __getattr__(name):
    return getattr(_real, name)

def __dir__():
    return dir(_real)

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
