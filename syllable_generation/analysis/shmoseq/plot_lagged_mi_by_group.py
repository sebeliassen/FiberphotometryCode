"""
[DEPRECATED SHIM] syllable_generation/analysis/shmoseq/plot_lagged_mi_by_group.py
==================================================
Relocated to: studies/behavior/shmoseq/plot_lagged_mi_by_group.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.analysis.shmoseq.plot_lagged_mi_by_group is deprecated; import from studies.behavior.shmoseq.plot_lagged_mi_by_group instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.shmoseq.plot_lagged_mi_by_group")
sys.modules[__name__] = _real

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
