"""
[DEPRECATED SHIM] syllable_generation/analysis/sequence_viz/probabilistic_episode_pilot/prepare_episode_stream.py
==================================================
Relocated to: studies/behavior/sequence_viz/probabilistic_episode_pilot/prepare_episode_stream.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.analysis.sequence_viz.probabilistic_episode_pilot.prepare_episode_stream is deprecated; import from studies.behavior.sequence_viz.probabilistic_episode_pilot.prepare_episode_stream instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.sequence_viz.probabilistic_episode_pilot.prepare_episode_stream")
sys.modules[__name__] = _real

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
