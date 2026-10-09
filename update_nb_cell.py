"""
[DEPRECATED SHIM] update_nb_cell.py
==================================================
Relocated to: ops/scripts/update_nb_cell.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "update_nb_cell is deprecated; import from ops.scripts.update_nb_cell instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("ops.scripts.update_nb_cell")
sys.modules[__name__] = _real

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
