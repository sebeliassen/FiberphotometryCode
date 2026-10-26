"""Historical import path for studies.kinematics_ulrik.lib.metrics."""

import importlib
import sys

_real = importlib.import_module("studies.kinematics_ulrik.lib.metrics")
sys.modules[__name__] = _real


def __getattr__(name):
    return getattr(_real, name)


def __dir__():
    return dir(_real)
