"""
[DEPRECATED SHIM] syllable_generation/analysis/sequence_viz/run_24_candidate_screening_batch.py
==================================================
Relocated to: studies/behavior/sequence_viz/run_24_candidate_screening_batch.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.analysis.sequence_viz.run_24_candidate_screening_batch is deprecated; import from studies.behavior.sequence_viz.run_24_candidate_screening_batch instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.sequence_viz.run_24_candidate_screening_batch")
sys.modules[__name__] = _real

def __getattr__(name):
    return getattr(_real, name)

def __dir__():
    return dir(_real)

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
