"""
[DEPRECATED SHIM] syllable_generation/analysis/sequence_viz/optimized_seqnmf/convolutional_seqnmf.py
==================================================
Relocated to: studies/behavior/sequence_viz/optimized_seqnmf/convolutional_seqnmf.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.analysis.sequence_viz.optimized_seqnmf.convolutional_seqnmf is deprecated; import from studies.behavior.sequence_viz.optimized_seqnmf.convolutional_seqnmf instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.sequence_viz.optimized_seqnmf.convolutional_seqnmf")
sys.modules[__name__] = _real

def __getattr__(name):
    return getattr(_real, name)

def __dir__():
    return dir(_real)

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
