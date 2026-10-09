"""
[DEPRECATED SHIM] scripts/ram_watchdog.py
==================================================
Relocated to: ops/scripts/ram_watchdog.py
"""

import importlib
import sys
import warnings

warnings.warn(
    "scripts.ram_watchdog is deprecated; import from ops.scripts.ram_watchdog instead",
    DeprecationWarning,
    stacklevel=2,
)

_real = importlib.import_module("ops.scripts.ram_watchdog")
sys.modules[__name__] = _real

if __name__ == "__main__":
    if hasattr(_real, "main"):
        _real.main()
