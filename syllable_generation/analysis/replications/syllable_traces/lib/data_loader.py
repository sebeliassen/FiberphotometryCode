"""
[DEPRECATED SHIM] syllable_generation/analysis/replications/syllable_traces/lib/data_loader.py
==================================================
Relocated to: studies/behavior/replications_datta/syllable_traces/lib/data_loader.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.analysis.replications.syllable_traces.lib.data_loader is deprecated; import from studies.behavior.replications_datta.syllable_traces.lib.data_loader instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.replications_datta.syllable_traces.lib.data_loader")
sys.modules[__name__] = _real

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
