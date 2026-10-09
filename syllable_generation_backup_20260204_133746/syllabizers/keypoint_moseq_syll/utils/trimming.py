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

def filter_with_masks(coordinates, confidences, masks, project_dir,
                      return_video_frame_indexes=False, 
                      downsample_rate=1, **kwargs):
    """
    Compress by masks (keep only mask==1 frames), run kpms.outlier_removal on compressed dicts.
    
    If downsample_rate > 1: 
    1. Slices data to [::downsample_rate] BEFORE filtering and outlier removal.
    2. Corrects video_frame_indexes to point to absolute frames in original video.
    """
    import numpy as np
    import keypoint_moseq as kpms

    names = set(coordinates) & set(confidences) & set(masks)
    if not names:
        raise ValueError("No overlapping keys among coordinates/confidences/masks.")

    # Apply downsampling if needed
    if downsample_rate > 1:
        coordinates = {k: v[::downsample_rate] for k, v in coordinates.items()}
        confidences = {k: v[::downsample_rate] for k, v in confidences.items()}
        # Note: masks must also be downsampled if they were calculated on high-res data.
        # In current pipeline, masks are calculated AFTER downsampling anyway.
        masks = {k: v[::downsample_rate] for k, v in masks.items()}

    for name in names:
        T = coordinates[name].shape[0]
        if confidences[name].shape[0] != T:
            raise ValueError(f"{name}: T mismatch conf vs coord")
        m = np.asarray(masks[name], bool)
        if m.shape != (T,):
            raise ValueError(f"{name}: mask must be shape ({T},), got {m.shape}")

    keep = {k: np.flatnonzero(np.asarray(masks[k], bool)) for k in names}

    coords_cmp = {k: coordinates[k][keep[k]] if keep[k].size else coordinates[k][:0] for k in names}
    confs_cmp  = {k: confidences[k][keep[k]] if keep[k].size else confidences[k][:0] for k in names}

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
        k: (coords_out[k] if k in coords_out else coordinates[k][:0].astype(float, copy=False))
        for k in names
    }
    filtered_confidences = {
        k: (confs_out[k] if k in confs_out else confidences[k][:0].astype(float, copy=False))
        for k in names
    }

    if return_video_frame_indexes:
        # Ground Truth Mapping: Map back to original indices by multiplying by downsample_rate
        video_frame_indexes = {
            k: (keep[k] * downsample_rate).astype(int) 
            for k in names
        }
        return filtered_coordinates, filtered_confidences, video_frame_indexes

    return filtered_coordinates, filtered_confidences

def clamp_conf_nonneg(conf_dict):
    """In-place clamp of confidences to [0, ∞)."""
    import numpy as np
    for k, v in conf_dict.items():
        np.maximum(v, 0.0, out=v)


def subset_spread_blocks(length, frac=0.1, n_blocks=5, block_len=None, seed=None):
    """
    Build a 1D index into a kept/filtered sequence (length=T) selecting ~frac*T
    frames as several contiguous 'slivers' spread over time (no center bias).

    Returns: np.ndarray of sorted, unique indices (int) into [0..T-1].
    """
    import numpy as np

    T = int(length)
    if T <= 0 or frac <= 0:
        return np.zeros((0,), dtype=int)

    target = max(1, int(round(frac * T)))
    if n_blocks <= 0:
        n_blocks = 1

    # If block_len not given, pick it so that n_blocks * block_len ~= target
    if block_len is None:
        block_len = max(1, target // n_blocks)

    # If the blocks would exceed T, shrink block_len
    block_len = min(block_len, T)

    # Spread starts uniformly, then (optionally) jitter to avoid determinism
    starts = np.linspace(0, max(0, T - block_len), num=n_blocks, dtype=int)
    if seed is not None and T > block_len:
        rng = np.random.default_rng(seed)
        max_jit = max(1, (T - block_len) // (n_blocks + 1))
        jitter = rng.integers(0, max_jit, size=n_blocks)
        starts = np.minimum(starts + jitter, T - block_len)

    idxs = []
    for s in starts:
        if len(idxs) >= target:
            break
        e = int(min(T, s + block_len))
        idxs.extend(range(s, e))

    # De-dup, sort, and cap to the target count
    idx = np.array(sorted(set(idxs)), dtype=int)
    if idx.size > target:
        idx = idx[:target]
    return idx


def subset_after_filter(filtered_coordinates, filtered_confidences, video_frame_indexes,
                        frac=0.1, n_blocks=5, block_len=None, seed=None):
    """
    Apply `subset_spread_blocks` per recording AFTER trimming/outlier removal.
    Returns new (coords, confs, vfi) dictionaries with only the subset frames.
    """
    import numpy as np

    sub_coords, sub_confs, sub_vfi = {}, {}, {}
    for k in filtered_coordinates.keys():
        T = len(filtered_coordinates[k])
        if T == 0:
            sub_coords[k] = filtered_coordinates[k][:0]
            sub_confs[k]  = filtered_confidences[k][:0]
            sub_vfi[k]    = video_frame_indexes[k][:0]
            continue

        idx = subset_spread_blocks(T, frac=frac, n_blocks=n_blocks,
                                   block_len=block_len, seed=seed)
        sub_coords[k] = filtered_coordinates[k][idx]
        sub_confs[k]  = filtered_confidences[k][idx]
        sub_vfi[k]    = video_frame_indexes[k][idx]

    return sub_coords, sub_confs, sub_vfi
