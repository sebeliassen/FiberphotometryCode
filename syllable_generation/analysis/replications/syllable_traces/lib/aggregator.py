"""
[DEPRECATED SHIM] syllable_generation/analysis/replications/syllable_traces/lib/aggregator.py
==================================================
Relocated to: studies/behavior/replications_datta/syllable_traces/lib/aggregator.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.analysis.replications.syllable_traces.lib.aggregator is deprecated; import from studies.behavior.replications_datta.syllable_traces.lib.aggregator instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.replications_datta.syllable_traces.lib.aggregator")
sys.modules[__name__] = _real

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
