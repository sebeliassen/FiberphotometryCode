"""Historical import facade for fpcode.behavior.primitives.resolutions.

Implementation lives in the canonical behavioural layer; this module
retains original function/class identities and private helper imports.
"""

from fpcode.behavior.primitives.resolutions import (
    member_composition as member_composition,
)
from fpcode.behavior.primitives.resolutions import (
    remap_states as remap_states,
)
from fpcode.behavior.primitives.resolutions import (
    standardised_member_execution as standardised_member_execution,
)
from fpcode.behavior.primitives.resolutions import (
    validate_state_map as validate_state_map,
)

__all__ = ['validate_state_map', 'remap_states', 'member_composition', 'standardised_member_execution']
