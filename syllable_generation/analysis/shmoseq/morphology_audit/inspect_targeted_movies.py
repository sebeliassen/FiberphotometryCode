"""
[DEPRECATED SHIM] syllable_generation/analysis/shmoseq/morphology_audit/inspect_targeted_movies.py
==================================================
Relocated to: studies/behavior/shmoseq/morphology_audit/inspect_targeted_movies.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.analysis.shmoseq.morphology_audit.inspect_targeted_movies is deprecated; import from studies.behavior.shmoseq.morphology_audit.inspect_targeted_movies instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.shmoseq.morphology_audit.inspect_targeted_movies")
sys.modules[__name__] = _real

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
