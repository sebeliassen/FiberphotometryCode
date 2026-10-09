"""
[DEPRECATED SHIM] syllable_generation/analysis/sequence_viz/optimized_seqnmf/inspect_multi_layer_durations.py
==================================================
Relocated to: studies/behavior/sequence_viz/optimized_seqnmf/inspect_multi_layer_durations.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.analysis.sequence_viz.optimized_seqnmf.inspect_multi_layer_durations is deprecated; import from studies.behavior.sequence_viz.optimized_seqnmf.inspect_multi_layer_durations instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.sequence_viz.optimized_seqnmf.inspect_multi_layer_durations")
sys.modules[__name__] = _real

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
