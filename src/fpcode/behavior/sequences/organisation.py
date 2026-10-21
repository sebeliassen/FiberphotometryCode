"""Historical import facade for fpcode.behavior.analysis.organisation.

Implementation lives in the canonical behavioural layer; this module
retains original function/class identities and private helper imports.
"""

from fpcode.behavior.analysis.organisation import (
    eligible_pairs as eligible_pairs,
)
from fpcode.behavior.analysis.organisation import (
    exclusive_timeline as exclusive_timeline,
)
from fpcode.behavior.analysis.organisation import (
    metric_values as metric_values,
)
from fpcode.behavior.analysis.organisation import (
    organisation_arrays as organisation_arrays,
)
from fpcode.behavior.analysis.organisation import (
    organisation_metrics as organisation_metrics,
)
from fpcode.behavior.analysis.organisation import (
    pair_audit as pair_audit,
)
from fpcode.behavior.analysis.organisation import (
    repetition_runs as repetition_runs,
)
from fpcode.behavior.analysis.organisation import (
    return_masks as return_masks,
)
from fpcode.behavior.analysis.organisation import (
    valid_frame_intervals as valid_frame_intervals,
)

__all__ = ['valid_frame_intervals', 'exclusive_timeline', 'organisation_arrays', 'eligible_pairs', 'pair_audit', 'return_masks', 'repetition_runs', 'organisation_metrics', 'metric_values']
