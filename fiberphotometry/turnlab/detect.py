def make_turn_syllable_series(
    df,
    session_idx_col='session_idx',
    frame_col='frame_index',
    omega_col='angular_velocity',
    speed_col='velocity_px_s',
    method='omega',                 # 'omega' or 'kappa'
    granularity_percentiles=None,   # e.g., [50] or [50, 90]; default [50]
    fps=80,
    # hysteresis / bout rules
    turn_start_pct=90,              # percentile of metric to start a turn
    stop_ratio=0.6,                 # stop = start * stop_ratio
    min_bout_ms=120,
    merge_gap_ms=120,
    # denoising / outlier handling
    smooth_win=5,                   # moving average window (frames); 1=off
    clip_hi_pct=99.5,               # winsorize high spikes per session; None to disable
    # curvature specifics
    speed_floor=2.0,                # px/s (or your units); below this, use |omega| to avoid kappa blow-up
    eps=1e-6
):
    """
    Returns:
      turn_syllable : pd.Series of str (e.g., 'left_50', 'right_90', NaN outside bouts)
      turn_onset    : pd.Series of bool (True at bout start, else False)
    """
    if granularity_percentiles is None:
        granularity_percentiles = [50]
    granularity_percentiles = sorted(granularity_percentiles)

    n = len(df)
    out_label = [None] * n
    out_onset = [False] * n

    # convenience
    def moving_avg(x, k):
        if k <= 1:
            return list(x)
        pad = k // 2
        seq = [x[0]]*pad + list(x) + [x[-1]]*pad
        sm = []
        for i in range(len(x)):
            sm.append(sum(seq[i:i+k]) / float(k))
        return sm

    def percentile(vals, p):
        s = sorted(vals)
        if not s:
            return 0.0
        idx = int(round((p/100.0)*(len(s)-1)))
        if idx < 0: idx = 0
        if idx >= len(s): idx = len(s)-1
        return s[idx]

    def to_frames(ms): 
        return int(round((ms * fps) / 1000.0))

    min_bout_frames = to_frames(min_bout_ms)
    merge_gap_frames = to_frames(merge_gap_ms)

    # ensure processing is in session order then frame order
    # (doesn't modify df; we work with indices)
    order = df.sort_values([session_idx_col, frame_col]).index.tolist()

    # prepare easy access columns
    sess_ids = df[session_idx_col].values
    omega_raw = df[omega_col].values
    speed_raw = df[speed_col].values if speed_col in df.columns else [None]*n

    # iterate by session
    i0 = 0
    while i0 < n:
        sid = sess_ids[order[i0]]
        # find contiguous block with same session_idx
        j = i0
        while j < n and sess_ids[order[j]] == sid:
            j += 1
        idxs = order[i0:j]                 # row indices for this session in order
        m = len(idxs)

        # pull series for this session
        omega = [float(omega_raw[k]) for k in idxs]
        speed = [float(speed_raw[k]) if speed_raw[k] is not None else 0.0 for k in idxs]

        # denoise: smooth
        omega_sm = moving_avg(omega, smooth_win)

        # denoise: winsorize high spikes (per session)
        if clip_hi_pct is not None:
            hi = percentile([abs(x) for x in omega_sm], clip_hi_pct)
            if hi > 0:
                omega_sm = [min(max(v, -hi), hi) for v in omega_sm]

        # build metric M for turn detection
        #  - omega-based: M = |omega_sm|
        #  - kappa-based: M = |omega_sm|/v for v >= speed_floor, else |omega_sm|
        absw = [abs(v) for v in omega_sm]
        if method == 'kappa':
            M = []
            for v_abs, v in zip(absw, speed):
                if v >= speed_floor:
                    M.append(v_abs / (v if v > eps else eps))
                else:
                    M.append(v_abs)  # in-place: fall back to |omega|
        else:
            M = absw

        # compute start/stop thresholds from session distribution
        start_thr = percentile([x for x in M], turn_start_pct)
        stop_thr = start_thr * stop_ratio

        # optional: turn granularity thresholds (percentiles of M)
        gran_cuts = [percentile(M, p) for p in granularity_percentiles]

        # hysteresis pass to label turning frames (state machine)
        turning = [False] * m
        on = False
        for t in range(m):
            if not on and M[t] >= start_thr:
                on = True
            elif on and M[t] <= stop_thr:
                on = False
            turning[t] = on

        # collect contiguous bouts
        bouts = []
        t = 0
        while t < m:
            if not turning[t]:
                t += 1
                continue
            s = t
            while t < m and turning[t]:
                t += 1
            e = t - 1
            bouts.append([s, e])

        # merge tiny gaps
        merged = []
        for b in bouts:
            if not merged:
                merged.append(b)
            else:
                if b[0] - merged[-1][1] - 1 <= merge_gap_frames:
                    merged[-1][1] = b[1]
                else:
                    merged.append(b)

        # drop short bouts
        bouts = [b for b in merged if (b[1] - b[0] + 1) >= min_bout_frames]

        # label bouts
        for s, e in bouts:
            # pick peak within bout on metric M
            peak_rel = max(range(e - s + 1), key=lambda k: M[s + k])
            peak = s + peak_rel
            # direction from smoothed omega at peak
            sign = 1 if omega_sm[peak] >= 0 else -1
            dir_str = 'right' if sign >= 0 else 'left'

            # assign granularity bin by peak metric
            peak_val = M[peak]
            chosen_p = None
            for p, thr in zip(granularity_percentiles[::-1], gran_cuts[::-1]):  # high->low
                if peak_val >= thr:
                    chosen_p = p
                    break
            if chosen_p is None:
                # if it passed start_thr it should usually exceed the lowest granularity cut;
                # but just in case, fall back to the lowest provided percentile
                chosen_p = granularity_percentiles[0]

            label = f"{dir_str}_{int(chosen_p)}"

            # write label across the whole bout; mark onset at s
            for t in range(s, e + 1):
                out_label[idxs[t]] = label
            out_onset[idxs[s]] = True

        # advance to next session block
        i0 = j

    # build Series with original index order
    turn_syllable = df[session_idx_col].copy()
    turn_syllable[:] = out_label
    turn_syllable.name = 'turn_syllable'

    turn_onset = df[session_idx_col].copy()
    turn_onset[:] = out_onset
    turn_onset = turn_onset.astype(bool)
    turn_onset.name = 'turn_onset'

    return turn_syllable, turn_onset
