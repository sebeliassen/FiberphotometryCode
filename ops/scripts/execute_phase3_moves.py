"""Compatibility alias for ops.repo.execute_phase3_moves."""
import importlib
import runpy
import sys
import warnings
from pathlib import Path

warnings.warn("ops.scripts.execute_phase3_moves is deprecated; use ops.repo.execute_phase3_moves",
              DeprecationWarning, stacklevel=2)

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "repo/execute_phase3_moves.py"), run_name="__main__")
else:
    _real = importlib.import_module("ops.repo.execute_phase3_moves")
    sys.modules[__name__] = _real

    def __getattr__(name):
        return getattr(_real, name)

    def __dir__():
        return dir(_real)
