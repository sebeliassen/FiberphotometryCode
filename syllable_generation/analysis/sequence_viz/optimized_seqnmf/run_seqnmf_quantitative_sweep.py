"""
[DEPRECATED SHIM] syllable_generation/analysis/sequence_viz/optimized_seqnmf/run_seqnmf_quantitative_sweep.py
==================================================
Relocated to: studies/behavior/sequence_viz/optimized_seqnmf/run_seqnmf_quantitative_sweep.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.analysis.sequence_viz.optimized_seqnmf.run_seqnmf_quantitative_sweep is deprecated; import from studies.behavior.sequence_viz.optimized_seqnmf.run_seqnmf_quantitative_sweep instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.sequence_viz.optimized_seqnmf.run_seqnmf_quantitative_sweep")
sys.modules[__name__] = _real

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
