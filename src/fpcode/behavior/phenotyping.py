"""Historical import facade for fpcode.behavior.analysis.phenotyping.

Implementation lives in the canonical behavioural layer; this module
retains original function/class identities and private helper imports.
"""

from fpcode.behavior.analysis.phenotyping import (
    _finite_stat as _finite_stat,
)
from fpcode.behavior.analysis.phenotyping import (
    analyze_frames as analyze_frames,
)
from fpcode.behavior.analysis.phenotyping import (
    diversity as diversity,
)
from fpcode.behavior.analysis.phenotyping import (
    frame_kinematics as frame_kinematics,
)

__all__ = ['diversity', 'frame_kinematics', 'analyze_frames']
