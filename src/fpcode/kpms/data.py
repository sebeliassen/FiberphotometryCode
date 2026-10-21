"""Historical import path; the canonical module retains the same objects."""

import importlib
import sys

sys.modules[__name__] = importlib.import_module('fpcode.kpms.preprocessing')
