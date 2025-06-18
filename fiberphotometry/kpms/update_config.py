import keypoint_moseq as kpms
import numpy as np

def update_config(project_dir, bodyparts):
    # Suppose cfg is your original configuration loaded from disk.
    cfg = kpms.load_config(project_dir)

    # Create a new list excluding tail parts.
    import re
    bodyparts_no_tail = [bp for bp in cfg['bodyparts'] if not re.match(r'tail\d$', bp)]

    # Create an updated skeleton by filtering out edges that reference removed bodyparts.
    new_skeleton = [edge for edge in cfg['skeleton'] if all(bp in bodyparts_no_tail for bp in edge)]

    # Recalculate indices (assuming anterior and posterior bodyparts remain the same).
    new_anterior_idxs = np.array([bodyparts_no_tail.index(bp)
                                for bp in cfg.get('anterior_bodyparts', [])
                                if bp in bodyparts_no_tail])
    new_posterior_idxs = np.array([bodyparts_no_tail.index(bp)
                                for bp in cfg.get('posterior_bodyparts', [])
                                if bp in bodyparts_no_tail])

    new_mask = np.array([bp in bodyparts_no_tail for bp in bodyparts])

    # Now update the configuration.
    kpms.update_config(
        project_dir,
        bodyparts=bodyparts_no_tail,
        use_bodyparts=bodyparts_no_tail,
        skeleton=new_skeleton,
        anterior_idxs=new_anterior_idxs,
        posterior_idxs=new_posterior_idxs
    )

    return new_mask