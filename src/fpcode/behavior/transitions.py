"""Historical import facade for fpcode.behavior.analysis.transitions.

Implementation lives in the canonical behavioural layer; this module
retains original function/class identities and private helper imports.
"""

from fpcode.behavior.analysis.transitions import (
    adjacent_pairs as adjacent_pairs,
)
from fpcode.behavior.analysis.transitions import (
    compute_stationary_distribution as compute_stationary_distribution,
)
from fpcode.behavior.analysis.transitions import (
    compute_transition_matrix as compute_transition_matrix,
)

__all__ = ['adjacent_pairs', 'compute_transition_matrix', 'compute_stationary_distribution']
