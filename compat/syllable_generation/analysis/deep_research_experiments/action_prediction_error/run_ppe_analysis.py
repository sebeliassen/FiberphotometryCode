"""
[DEPRECATED SHIM] syllable_generation/analysis/deep_research_experiments/action_prediction_error/run_ppe_analysis.py
==================================================
Relocated to: studies/deep_research_experiments/action_prediction_error/run_ppe_analysis.py
"""

import importlib
import sys
import warnings
from pathlib import Path

warnings.warn(
    "syllable_generation.analysis.deep_research_experiments.action_prediction_error.run_ppe_analysis is deprecated; import from studies.deep_research_experiments.action_prediction_error.run_ppe_analysis instead",
    DeprecationWarning,
    stacklevel=2,
)

_bootstrap_paths = []
if __name__ == "__main__" and not __package__:
    _repo_root = Path(__file__).resolve().parents[5]
    for _source_dir in (_repo_root, _repo_root / "src"):
        if str(_source_dir) not in sys.path:
            sys.path.insert(0, str(_source_dir))
            _bootstrap_paths.append(str(_source_dir))

try:
    _real = importlib.import_module("studies.deep_research_experiments.action_prediction_error.run_ppe_analysis")
except BaseException:
    for _source_dir in _bootstrap_paths:
        if _source_dir in sys.path:
            sys.path.remove(_source_dir)
    raise
sys.modules[__name__] = _real

def __getattr__(name):
    return getattr(_real, name)

def __dir__():
    return dir(_real)

if __name__ == "__main__":
    try:
        if hasattr(_real, "main"):
            _real.main()
    finally:
        for _source_dir in _bootstrap_paths:
            if _source_dir in sys.path:
                sys.path.remove(_source_dir)
