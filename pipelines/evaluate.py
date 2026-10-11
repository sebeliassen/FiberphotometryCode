"""Compatibility entrypoint; canonical implementation is fpcode.cli.evaluate."""
import sys

from fpcode.cli import evaluate as _canonical

if __name__ == "__main__":
    raise SystemExit(_canonical.main())

sys.modules[__name__] = _canonical
