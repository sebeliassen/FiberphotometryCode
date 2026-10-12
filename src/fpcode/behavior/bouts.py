"""Historical import facade for fpcode.behavior.primitives.bouts.

Implementation lives in the canonical behavioural layer; this module
retains original function/class identities and private helper imports.
"""

from fpcode.behavior.primitives.bouts import (
    _contiguous_rows as _contiguous_rows,
)
from fpcode.behavior.primitives.bouts import (
    extract_bouts as extract_bouts,
)
from fpcode.behavior.primitives.bouts import (
    extract_syllable_runs as extract_syllable_runs,
)
from fpcode.behavior.primitives.bouts import (
    run_boundaries as run_boundaries,
)

__all__ = ['run_boundaries', 'extract_bouts', 'extract_syllable_runs']
