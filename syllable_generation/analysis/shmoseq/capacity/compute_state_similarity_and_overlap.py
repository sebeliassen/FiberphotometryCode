"""
[DEPRECATED SHIM] syllable_generation/analysis/shmoseq/capacity/compute_state_similarity_and_overlap.py
==================================================
Relocated to: studies/behavior/shmoseq/capacity/compute_state_similarity_and_overlap.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.analysis.shmoseq.capacity.compute_state_similarity_and_overlap is deprecated; import from studies.behavior.shmoseq.capacity.compute_state_similarity_and_overlap instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.shmoseq.capacity.compute_state_similarity_and_overlap")
sys.modules[__name__] = _real

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
