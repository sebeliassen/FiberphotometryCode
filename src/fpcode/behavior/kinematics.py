"""Historical import facade for fpcode.behavior.observations.kinematics.

Implementation lives in the canonical behavioural layer; this module
retains original function/class identities and private helper imports.
"""

from fpcode.behavior.observations.kinematics import (
    clean_short_runs as clean_short_runs,
)
from fpcode.behavior.observations.kinematics import (
    detect_movement_initiations as detect_movement_initiations,
)
from fpcode.behavior.observations.kinematics import (
    legacy_nose_tail_kinematics as legacy_nose_tail_kinematics,
)
from fpcode.behavior.observations.kinematics import (
    movement_changepoint_score as movement_changepoint_score,
)
from fpcode.behavior.observations.kinematics import (
    presence_mask_from_conf as presence_mask_from_conf,
)
from fpcode.behavior.observations.kinematics import (
    presence_mask_from_conf_kpms as presence_mask_from_conf_kpms,
)
from fpcode.behavior.observations.kinematics import (
    smooth_centroid_median as smooth_centroid_median,
)
from fpcode.behavior.observations.kinematics import (
    smooth_centroid_median_legacy_fusion as smooth_centroid_median_legacy_fusion,
)
from fpcode.behavior.observations.kinematics import (
    smooth_heading_circular as smooth_heading_circular,
)
from fpcode.behavior.observations.kinematics import (
    smooth_heading_circular_legacy_fusion as smooth_heading_circular_legacy_fusion,
)
from fpcode.behavior.observations.kinematics import (
    subset_by_activity as subset_by_activity,
)

__all__ = ['movement_changepoint_score', 'detect_movement_initiations', 'legacy_nose_tail_kinematics', 'smooth_heading_circular', 'smooth_centroid_median', 'clean_short_runs', 'presence_mask_from_conf', 'subset_by_activity', 'smooth_heading_circular_legacy_fusion', 'smooth_centroid_median_legacy_fusion', 'presence_mask_from_conf_kpms']
