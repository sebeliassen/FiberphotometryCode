"""
[DEPRECATED SHIM] syllable_generation/analysis/sequence_viz/hierarchical_pilot/run_controlled_upper_timescale_experiment.py
==================================================
Relocated to: studies/behavior/sequence_viz/hierarchical_pilot/run_controlled_upper_timescale_experiment.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.analysis.sequence_viz.hierarchical_pilot.run_controlled_upper_timescale_experiment is deprecated; import from studies.behavior.sequence_viz.hierarchical_pilot.run_controlled_upper_timescale_experiment instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.sequence_viz.hierarchical_pilot.run_controlled_upper_timescale_experiment")
sys.modules[__name__] = _real

def __getattr__(name):
    return getattr(_real, name)

def __dir__():
    return dir(_real)

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
