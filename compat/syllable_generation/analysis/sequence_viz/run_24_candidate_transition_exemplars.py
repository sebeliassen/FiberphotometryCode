"""
[DEPRECATED SHIM] syllable_generation/analysis/sequence_viz/run_24_candidate_transition_exemplars.py
==================================================
Relocated to: studies/behavior/sequence_viz/run_24_candidate_transition_exemplars.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.analysis.sequence_viz.run_24_candidate_transition_exemplars is deprecated; import from studies.behavior.sequence_viz.run_24_candidate_transition_exemplars instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.sequence_viz.run_24_candidate_transition_exemplars")
sys.modules[__name__] = _real

def __getattr__(name):
    return getattr(_real, name)

def __dir__():
    return dir(_real)

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
