"""Historical import facade for fpcode.behavior.episodes.seqnmf.

Implementation lives in the canonical behavioural layer; this module
retains original function/class identities and private helper imports.
"""

from fpcode.behavior.episodes.seqnmf import (
    SeqNMF as SeqNMF,
)
from fpcode.behavior.episodes.seqnmf import (
    _adjoint as _adjoint,
)
from fpcode.behavior.episodes.seqnmf import (
    _smooth as _smooth,
)
from fpcode.behavior.episodes.seqnmf import (
    activation_events as activation_events,
)
from fpcode.behavior.episodes.seqnmf import (
    reconstruct as reconstruct,
)
from fpcode.behavior.episodes.seqnmf import (
    template_similarity as template_similarity,
)

__all__ = ['reconstruct', 'SeqNMF', 'template_similarity', 'activation_events']
