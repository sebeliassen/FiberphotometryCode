"""
[DEPRECATED SHIM] scripts/batch_compact_all_checkpoints.py
==================================================
Relocated to: ops/scripts/batch_compact_all_checkpoints.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "scripts.batch_compact_all_checkpoints is deprecated; import from ops.scripts.batch_compact_all_checkpoints instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("ops.scripts.batch_compact_all_checkpoints")
sys.modules[__name__] = _real

def __getattr__(name):
    return getattr(_real, name)

def __dir__():
    return dir(_real)

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
