"""Historical I/O alias; generic I/O and KPMS checkpoint I/O have separate owners."""

import importlib
import sys

_real = importlib.import_module("fpcode.compat.io")
sys.modules[__name__] = _real


def __getattr__(name):
    return getattr(_real, name)


def __dir__():
    return dir(_real)
