"""
[DEPRECATED SHIM] scripts/execute_phase3_moves.py
==================================================
Relocated to: ops/scripts/execute_phase3_moves.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "scripts.execute_phase3_moves is deprecated; import from ops.scripts.execute_phase3_moves instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("ops.scripts.execute_phase3_moves")
sys.modules[__name__] = _real

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
