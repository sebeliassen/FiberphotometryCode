"""Historical import facade for fpcode.behavior.episodes.chains.

Implementation lives in the canonical behavioural layer; this module
retains original function/class identities and private helper imports.
"""

from fpcode.behavior.episodes.chains import (
    edit_alignment as edit_alignment,
)
from fpcode.behavior.episodes.chains import (
    exact_occurrences as exact_occurrences,
)
from fpcode.behavior.episodes.chains import (
    group_occurrences as group_occurrences,
)
from fpcode.behavior.episodes.chains import (
    ngram_counts as ngram_counts,
)
from fpcode.behavior.episodes.chains import (
    prefix_progression as prefix_progression,
)
from fpcode.behavior.episodes.chains import (
    segmented_tokens as segmented_tokens,
)

__all__ = ['segmented_tokens', 'ngram_counts', 'edit_alignment', 'exact_occurrences', 'prefix_progression', 'group_occurrences']
