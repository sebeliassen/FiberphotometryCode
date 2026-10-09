
import numpy as np


def _clean_short_runs(mask, min_true=1, min_false=1):
    """Remove ON runs shorter than min_true, and OFF runs shorter than min_false."""
    m = mask.copy()
    if m.size == 0:
        return m

    x = m.astype(np.int8)
    # Run boundaries
    boundaries = np.flatnonzero(np.r_[True, x[1:] != x[:-1], True])
    starts = boundaries[:-1]
    ends = boundaries[1:]
    vals = m[starts]

    # First pass: kill short True runs
    if min_true > 1:
        for s, e, v in zip(starts, ends, vals):
            if v and (e - s) < min_true:
                m[s:e] = False

    # Recompute runs after first pass
    x = m.astype(np.int8)
    boundaries = np.flatnonzero(np.r_[True, x[1:] != x[:-1], True])
    starts = boundaries[:-1]
    ends = boundaries[1:]
    vals = m[starts]

    # Second pass: fill short False gaps
    if min_false > 1:
        for s, e, v in zip(starts, ends, vals):
            if (not v) and (e - s) < min_false:
                m[s:e] = True

    return m

def presence_mask_from_conf(conf,
                            p_thresh=0.9,          # bodypart counted as "seen" if >= this
                            frac_min_parts=0.4,    # min fraction of parts to seed presence
                            smooth_win=7,          # moving avg window (frames)
                            enter_thresh=0.65,     # hysteresis enter threshold (on smoothed fraction)
                            exit_thresh=0.45,      # hysteresis exit threshold
                            off_grace_frames=6,    # require this many bad frames in a row to exit
                            min_on_frames=10,      # remove ON runs shorter than this
                            min_off_frames=6):     # fill OFF gaps shorter than this
    T, N = conf.shape

    # 1) Per-frame fraction of confidently-seen bodyparts
    valid = (conf >= p_thresh)
    count = valid.sum(axis=1)
    frac = count / float(N)

    # 2) Light smoothing
    if smooth_win > 1:
        k = np.ones(smooth_win, float) / smooth_win
        sm = np.convolve(frac, k, mode="same")
    else:
        sm = frac

    # 3) Seed frames that clearly have enough parts (helps resist brief dips)
    k_min = max(1, int(np.ceil(frac_min_parts * N)))
    seed = (count >= k_min)

    # 4) Hysteresis with "grace" to avoid flipping OFF on brief occlusions
    on = np.zeros(T, dtype=bool)
    inside = False
    below = 0
    for t in range(T):
        if not inside:
            if (sm[t] >= enter_thresh) or seed[t]:
                inside = True
                below = 0
        else:
            if (sm[t] <= exit_thresh) and (not seed[t]):
                below += 1
                if below >= off_grace_frames:
                    inside = False
                    below = 0
            else:
                below = 0
        on[t] = inside

    # 5) Remove very short ON runs, then fill very short OFF gaps
    if min_on_frames > 1 or min_off_frames > 1:
        on = _clean_short_runs(on, min_true=min_on_frames, min_false=min_off_frames)

    return on

def filter_with_masks_and_downsample(coordinates, confidences, project_dir,
                      downsample_rate=1, return_video_frame_indexes=False, **kwargs):
    """
    Combined Downsampling + Filtering.
    """
    import keypoint_moseq as kpms
    import numpy as np

    names = set(coordinates) & set(confidences)
    if not names:
        raise ValueError("No overlapping keys among coordinates/confidences.")

    ds_coords = {}
    ds_confs = {}
    all_masks = {}

    for name in names:
        # 1. Downsample
        if downsample_rate > 1:
            ds_coords[name] = coordinates[name][::downsample_rate]
            ds_confs[name]  = confidences[name][::downsample_rate]
        else:
            ds_coords[name] = coordinates[name]
            ds_confs[name]  = confidences[name]

        # 2. Generate Mask (on downsampled data)
        # We extract only the relevant args for presence_mask_from_conf from kwargs
        mask_arg_names = [
            'p_thresh', 'frac_min_parts', 'smooth_win', 'enter_thresh',
            'exit_thresh', 'off_grace_frames', 'min_on_frames', 'min_off_frames'
        ]
        mask_kwargs = {k: v for k, v in kwargs.items() if k in mask_arg_names}
        all_masks[name] = presence_mask_from_conf(ds_confs[name], **mask_kwargs)

    # 3. Filtering
    keep = {k: np.flatnonzero(np.asarray(all_masks[k], bool)) for k in names}
    coords_cmp = {k: ds_coords[k][keep[k]] if keep[k].size else ds_coords[k][:0] for k in names}
    confs_cmp  = {k: ds_confs[k][keep[k]] if keep[k].size else ds_confs[k][:0] for k in names}

    nonempty = [k for k in names if keep[k].size]
    coords_out, confs_out = {}, {}
    if nonempty:
        coords_out, confs_out = kpms.outlier_removal(
            {k: coords_cmp[k] for k in nonempty},
            {k: confs_cmp[k]  for k in nonempty},
            project_dir,
            **kwargs
        )

    filtered_coordinates = {
        k: (coords_out[k] if k in coords_out else ds_coords[k][:0].astype(float, copy=False))
        for k in names
    }
    filtered_confidences = {
        k: (confs_out[k] if k in confs_out else ds_confs[k][:0].astype(float, copy=False))
        for k in names
    }

    if return_video_frame_indexes:
        # Map back to original indices
        video_frame_indexes = {
            k: (keep[k] * downsample_rate).astype(int)
            for k in names
        }
        return filtered_coordinates, filtered_confidences, video_frame_indexes

    return filtered_coordinates, filtered_confidences
