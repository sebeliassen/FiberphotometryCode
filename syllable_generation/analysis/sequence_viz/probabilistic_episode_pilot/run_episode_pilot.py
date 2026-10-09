"""
[DEPRECATED SHIM] syllable_generation/analysis/sequence_viz/probabilistic_episode_pilot/run_episode_pilot.py
==================================================
Relocated to: studies/behavior/sequence_viz/probabilistic_episode_pilot/run_episode_pilot.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.analysis.sequence_viz.probabilistic_episode_pilot.run_episode_pilot is deprecated; import from studies.behavior.sequence_viz.probabilistic_episode_pilot.run_episode_pilot instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.sequence_viz.probabilistic_episode_pilot.run_episode_pilot")
sys.modules[__name__] = _real

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
