"""Compatibility entrypoint; canonical implementation is fpcode.cli.render."""
import sys
from fpcode.cli import render as _canonical

if __name__ == "__main__":
    raise SystemExit(_canonical.main())

sys.modules[__name__] = _canonical
