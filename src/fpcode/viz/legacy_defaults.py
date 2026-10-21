"""Historical alias; cohort-specific implementation lives in fpcode.compat.viz."""

import importlib
import sys

_real = importlib.import_module("fpcode.compat.viz.legacy_defaults")
sys.modules[__name__] = _real


def __getattr__(name):
    return getattr(_real, name)


def __dir__():
    return dir(_real)
