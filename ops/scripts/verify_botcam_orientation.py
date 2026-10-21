"""Compatibility alias for ops.dlc.verify_botcam_orientation."""
import importlib
import runpy
import sys
import warnings
from pathlib import Path

warnings.warn("ops.scripts.verify_botcam_orientation is deprecated; use ops.dlc.verify_botcam_orientation",
              DeprecationWarning, stacklevel=2)

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "dlc/verify_botcam_orientation.py"), run_name="__main__")
else:
    _real = importlib.import_module("ops.dlc.verify_botcam_orientation")
    sys.modules[__name__] = _real

    def __getattr__(name):
        return getattr(_real, name)

    def __dir__():
        return dir(_real)
