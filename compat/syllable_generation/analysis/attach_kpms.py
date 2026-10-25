"""
[DEPRECATED SHIM] syllable_generation/analysis/attach_kpms.py
==================================================
Relocated to: studies/session_model_exports/attach_kpms.py
"""

import importlib
import sys
import warnings
from pathlib import Path

warnings.warn(
    "syllable_generation.analysis.attach_kpms is deprecated; import from studies.session_model_exports.attach_kpms instead",
    DeprecationWarning,
    stacklevel=2,
)

_bootstrap_paths = []
if __name__ == "__main__" and not __package__:
    _repo_root = Path(__file__).resolve().parents[3]
    for _source_dir in (_repo_root, _repo_root / "src"):
        if str(_source_dir) not in sys.path:
            sys.path.insert(0, str(_source_dir))
            _bootstrap_paths.append(str(_source_dir))

try:
    _real = importlib.import_module("studies.session_model_exports.attach_kpms")
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
