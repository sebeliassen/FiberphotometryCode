"""
[DEPRECATED SHIM] syllable_generation/analysis/replications/syllable_traces/run_trace_analysis.py
==================================================
Relocated to: studies/behavior/replications_datta/syllable_traces/run_trace_analysis.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.analysis.replications.syllable_traces.run_trace_analysis is deprecated; import from studies.behavior.replications_datta.syllable_traces.run_trace_analysis instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.replications_datta.syllable_traces.run_trace_analysis")
sys.modules[__name__] = _real

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
