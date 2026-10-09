"""
camera_correspondence.py [Compatibility Shim]
==========================================
Transparently proxies fpcode.tracking.camera_correspondence.
"""

import importlib
import sys

_real = importlib.import_module("fpcode.tracking.camera_correspondence")
sys.modules[__name__] = _real
