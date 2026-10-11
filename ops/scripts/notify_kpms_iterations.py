"""Compatibility alias for ops.monitoring.notify_kpms_iterations."""
import importlib
import runpy
import sys
import warnings
from pathlib import Path

warnings.warn("ops.scripts.notify_kpms_iterations is deprecated; use ops.monitoring.notify_kpms_iterations",
              DeprecationWarning, stacklevel=2)

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "monitoring/notify_kpms_iterations.py"), run_name="__main__")
else:
    _real = importlib.import_module("ops.monitoring.notify_kpms_iterations")
    sys.modules[__name__] = _real

    def __getattr__(name):
        return getattr(_real, name)

    def __dir__():
        return dir(_real)
