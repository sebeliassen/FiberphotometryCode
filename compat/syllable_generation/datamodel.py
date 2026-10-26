"""Historical import path for fpcode.contracts.session_record."""

import importlib
import sys

_real = importlib.import_module("fpcode.contracts.session_record")
sys.modules[__name__] = _real


def __getattr__(name):
    return getattr(_real, name)


def __dir__():
    return dir(_real)
