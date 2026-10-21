"""
viz_v2_common.py [Compatibility Shim]
==========================================
Transparently proxies fpcode.viz.v2.
"""
import importlib
import sys

_real = importlib.import_module("fpcode.viz.v2")
sys.modules[__name__] = _real

def __getattr__(name):
    return getattr(_real, name)

def __dir__():
    return dir(_real)
