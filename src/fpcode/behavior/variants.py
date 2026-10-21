"""Historical import facade for fpcode.behavior.analysis.variants.

Implementation lives in the canonical behavioural layer; this module
retains original function/class identities and private helper imports.
"""

from fpcode.behavior.analysis.variants import (
    assign_execution_variants as assign_execution_variants,
)
from fpcode.behavior.analysis.variants import (
    fit_execution_variants as fit_execution_variants,
)

__all__ = ['fit_execution_variants', 'assign_execution_variants']
