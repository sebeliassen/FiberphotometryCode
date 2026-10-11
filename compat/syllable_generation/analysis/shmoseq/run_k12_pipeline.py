"""
[DEPRECATED SHIM] syllable_generation/analysis/shmoseq/run_k12_pipeline.py
==================================================
Relocated to: studies/behavior/shmoseq/run_k12_pipeline.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.analysis.shmoseq.run_k12_pipeline is deprecated; import from studies.behavior.shmoseq.run_k12_pipeline instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.shmoseq.run_k12_pipeline")
sys.modules[__name__] = _real

def __getattr__(name):
    return getattr(_real, name)

def __dir__():
    return dir(_real)

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
