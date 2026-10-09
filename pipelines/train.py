"""Compatibility entrypoint; canonical implementation is fpcode.cli.train."""
import sys
from fpcode.cli import train as _canonical

if __name__ == "__main__":
    raise SystemExit(_canonical.main())

sys.modules[__name__] = _canonical
