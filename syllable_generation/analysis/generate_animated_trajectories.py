"""
[DEPRECATED SHIM] syllable_generation/analysis/generate_animated_trajectories.py
==================================================
Relocated to: studies/behavior/misc/syllable_generation/analysis/generate_animated_trajectories.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.analysis.generate_animated_trajectories is deprecated; import from studies.behavior.misc.syllable_generation.analysis.generate_animated_trajectories instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.misc.syllable_generation.analysis.generate_animated_trajectories")
sys.modules[__name__] = _real

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
