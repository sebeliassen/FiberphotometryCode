"""Compatibility alias for ops.monitoring.launch_extended_when_limited_done."""
import importlib
import runpy
import sys
import warnings
from pathlib import Path

warnings.warn("ops.scripts.launch_extended_when_limited_done is deprecated; use ops.monitoring.launch_extended_when_limited_done",
              DeprecationWarning, stacklevel=2)

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "monitoring/launch_extended_when_limited_done.py"), run_name="__main__")
else:
    _real = importlib.import_module("ops.monitoring.launch_extended_when_limited_done")
    sys.modules[__name__] = _real

    def __getattr__(name):
        return getattr(_real, name)

    def __dir__():
        return dir(_real)
