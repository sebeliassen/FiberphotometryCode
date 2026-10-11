"""Compatibility alias for ops.monitoring.ram_watchdog."""
import importlib
import runpy
import sys
import warnings
from pathlib import Path

warnings.warn("ops.scripts.ram_watchdog is deprecated; use ops.monitoring.ram_watchdog",
              DeprecationWarning, stacklevel=2)

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "monitoring/ram_watchdog.py"), run_name="__main__")
else:
    _real = importlib.import_module("ops.monitoring.ram_watchdog")
    sys.modules[__name__] = _real

    def __getattr__(name):
        return getattr(_real, name)

    def __dir__():
        return dir(_real)
