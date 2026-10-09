"""
[DEPRECATED SHIM] syllable_generation/analysis/shmoseq/run_exact_animal_permutation_test.py
==================================================
Relocated to: studies/behavior/shmoseq/run_exact_animal_permutation_test.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.analysis.shmoseq.run_exact_animal_permutation_test is deprecated; import from studies.behavior.shmoseq.run_exact_animal_permutation_test instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.shmoseq.run_exact_animal_permutation_test")
sys.modules[__name__] = _real

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
