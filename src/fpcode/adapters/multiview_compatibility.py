"""Historical multiview adapter facade; new callers pass explicit resources."""

import importlib
import sys

_real = importlib.import_module("fpcode.compat.multiview.profile")
sys.modules[__name__] = _real


def __getattr__(name):
    return getattr(_real, name)


def __dir__():
    return dir(_real)
