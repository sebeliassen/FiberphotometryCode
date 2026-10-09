"""
[DEPRECATED SHIM] syllable_generation/analysis/shmoseq/diagnostic_lagged_mi_bias.py
==================================================
Relocated to: studies/behavior/shmoseq/diagnostic_lagged_mi_bias.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.analysis.shmoseq.diagnostic_lagged_mi_bias is deprecated; import from studies.behavior.shmoseq.diagnostic_lagged_mi_bias instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.shmoseq.diagnostic_lagged_mi_bias")
sys.modules[__name__] = _real

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
