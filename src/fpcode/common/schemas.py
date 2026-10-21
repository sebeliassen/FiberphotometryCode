"""Historical schemas facade; algorithms import shapes from fpcode.contracts."""

import importlib
import sys

_real = importlib.import_module("fpcode.compat.schemas")
sys.modules[__name__] = _real


def __getattr__(name):
    return getattr(_real, name)


def __dir__():
    return dir(_real)
