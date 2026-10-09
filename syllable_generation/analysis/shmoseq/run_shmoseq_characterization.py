"""
[DEPRECATED SHIM] syllable_generation/analysis/shmoseq/run_shmoseq_characterization.py
==================================================
Relocated to: studies/behavior/shmoseq/run_shmoseq_characterization.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.analysis.shmoseq.run_shmoseq_characterization is deprecated; import from studies.behavior.shmoseq.run_shmoseq_characterization instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.shmoseq.run_shmoseq_characterization")
sys.modules[__name__] = _real

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
