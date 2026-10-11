"""
[DEPRECATED SHIM] syllable_generation/analysis/language_models/pipelines/01_horizon_dynamics/run_multi_horizon_benchmark.py
==================================================
Relocated to: studies/behavior/language_models/pipelines/01_horizon_dynamics/run_multi_horizon_benchmark.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "syllable_generation.analysis.language_models.pipelines.01_horizon_dynamics.run_multi_horizon_benchmark is deprecated; import from studies.behavior.language_models.pipelines.01_horizon_dynamics.run_multi_horizon_benchmark instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("studies.behavior.language_models.pipelines.01_horizon_dynamics.run_multi_horizon_benchmark")
sys.modules[__name__] = _real

def __getattr__(name):
    return getattr(_real, name)

def __dir__():
    return dir(_real)

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
