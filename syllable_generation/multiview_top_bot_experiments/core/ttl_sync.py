"""
ttl_sync.py [Compatibility Shim]
==========================================
Transparently proxies fpcode.tracking.clock_sync.
"""
import importlib
import sys

_real = importlib.import_module("fpcode.tracking.clock_sync")
sys.modules[__name__] = _real
