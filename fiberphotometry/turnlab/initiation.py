from tqdm import tqdm   # add this import at the top of the function

def make_initiation_series(
    df,
    session_idx_col='session_idx',
    frame_col='frame_index',
    speed_col='velocity_px_s',          # <- uses your speed column
    omega_col='angular_velocity',       # <- direction proxy (sign) if head_dir not available
    fps=80,
    min_rest_ms=400,                    # paper: ≥400 ms below threshold
    min_move_ms=120,                    # paper: ≥120 ms above threshold
    sd_mult=2.0,                        # threshold = max(sd_mult * SD(speed), speed_floor)
    speed_floor=25.0                    # mm/s (px/s in your data)
):
    """
    Returns (init_label, init_onset) aligned to df.index.
      - init_label: 'left_init' / 'right_init' at the onset frame (None elsewhere)
      - init_onset: True at rest→move onset, else False

    Uses only speed_col for rest→move; left/right from sign(omega_col).
    """
    import numpy as np, pandas as pd
    from numpy.lib.stride_tricks import sliding_window_view as swv

    init_label = pd.Series([None]*len(df), index=df.index, dtype='object', name='turn_syllable')
    init_onset = pd.Series(False, index=df.index, dtype=bool, name='turn_onset')

    order = df.sort_values([session_idx_col, frame_col]).index.to_numpy()
    sess  = df.loc[order, session_idx_col].to_numpy()
    spd   = df.loc[order, speed_col].to_numpy(dtype=float)
    have_omega = (omega_col in df.columns)
    omg = df.loc[order, omega_col].to_numpy(dtype=float) if have_omega else None

    k_rest = max(1, int(round(min_rest_ms * fps / 1000.0)))
    k_move = max(1, int(round(min_move_ms * fps / 1000.0)))

    i0 = 0
    n = len(order)
    while i0 < n:
        sid = sess[i0]
        j = i0
        while j < n and sess[j] == sid:
            j += 1

        idxs = order[i0:j]
        s = spd[i0:j]

        # session-specific threshold
        thr = max(sd_mult * float(np.nanstd(s)), float(speed_floor))

        # states
        rest = s < thr
        move = s > thr

        ok_rest = np.zeros_like(rest, dtype=bool)
        ok_move = np.zeros_like(move, dtype=bool)
        if len(rest) >= k_rest:
            ok_rest[k_rest-1:] = swv(rest, k_rest).all(axis=1)
        if len(move) >= k_move:
            ok_move[:len(move)-k_move+1] = swv(move, k_move).all(axis=1)

        # direction proxy from angular velocity sign (optional)
        if have_omega:
            o = omg[i0:j]
            lr = np.where(o >= 0.0, 'right', 'left')
        else:
            lr = np.array([None]*len(idxs), dtype=object)

        # scan for onsets
        t = 1
        while t < len(s):
            if ok_move[t] and ok_rest[max(0, t-1)] and not ok_move[t-1]:
                onset_df_i = int(idxs[t])
                init_onset.at[onset_df_i] = True
                d = lr[t]
                if d == 'left':
                    init_label.at[onset_df_i] = 'left_init'
                elif d == 'right':
                    init_label.at[onset_df_i] = 'right_init'
                else:
                    init_label.at[onset_df_i] = None

                # skip until we leave high-mobility to avoid duplicate onsets
                u = t + 1
                while u < len(s) and move[u]:
                    u += 1
                t = u
            else:
                t += 1

        i0 = j

    return init_label, init_onset

def sweep_initiation_params_simple(
    df,
    fps,
    session_idx_col='session_idx',
    frame_col='frame_index',
    speed_col='velocity_px_s',
    # --- search space (lists) ---
    min_rest_ms_list=(100, 200, 300, 400, 600),
    min_move_ms_list=(60, 80, 100, 120, 160),
    sd_mult_list=(0.8, 1.0, 1.25, 1.5, 2.0),
    speed_floor_list=(5.0, 10.0, 20.0, 25.0, 40.0),
    # --- options ---
    edge_window_s=None,        # e.g., 10.0 to count onsets with full ±10s inside session
    random_samples=None,       # int → sample this many combos uniformly at random
    random_seed=0
):
    """
    Count movement-initiation onsets across combos of the *original four* parameters.
    Returns a DataFrame sorted by N_onsets (desc). Columns:
      min_rest_ms, min_move_ms, sd_mult, speed_floor, N_onsets, N_edge_safe (if requested), sessions_with_onsets
    """
    import numpy as np, pandas as pd
    from numpy.lib.stride_tricks import sliding_window_view as swv
    from itertools import product

    # ---- Pre-sort once; build per-session slices ----
    order = df.sort_values([session_idx_col, frame_col]).index.to_numpy()
    sess  = df.loc[order, session_idx_col].to_numpy()
    spd   = df.loc[order, speed_col].to_numpy(dtype=float)

    sess_slices = {}
    i = 0
    n = len(order)
    while i < n:
        s = sess[i]
        j = i
        while j < n and sess[j] == s:
            j += 1
        sess_slices[int(s)] = (i, j)
        i = j

    # ---- Build combo list (grid or random) ----
    grid = list(product(min_rest_ms_list, min_move_ms_list, sd_mult_list, speed_floor_list))
    if random_samples is not None and random_samples < len(grid):
        rng = np.random.default_rng(random_seed)
        idx = rng.choice(len(grid), size=int(random_samples), replace=False)
        grid = [grid[k] for k in idx]

    def _count_session(s_arr, min_rest_ms, min_move_ms, sd_mult, speed_floor):
        k_rest = max(1, int(round(min_rest_ms * fps / 1000.0)))
        k_move = max(1, int(round(min_move_ms * fps / 1000.0)))
        if s_arr.size == 0:
            return 0, 0

        thr = max(float(sd_mult) * float(np.nanstd(s_arr)), float(speed_floor))
        rest = s_arr < thr
        move = s_arr > thr

        ok_rest = np.zeros_like(rest, dtype=bool)
        ok_move = np.zeros_like(move, dtype=bool)
        if rest.size >= k_rest:
            ok_rest[k_rest-1:] = swv(rest, k_rest).all(axis=1)
        if move.size >= k_move:
            ok_move[:move.size - k_move + 1] = swv(move, k_move).all(axis=1)

        prev_ok_move = np.zeros_like(ok_move, dtype=bool); prev_ok_move[1:] = ok_move[:-1]
        prev_ok_rest = np.zeros_like(ok_rest, dtype=bool); prev_ok_rest[1:] = ok_rest[:-1]
        starts = np.where(ok_move & (~prev_ok_move) & prev_ok_rest)[0]

        if edge_window_s is not None:
            w = int(round(edge_window_s * fps))
            edge_safe = int(np.sum((starts >= w) & (starts < (len(s_arr) - w))))
        else:
            edge_safe = 0

        return int(starts.size), edge_safe

    rows = []
    for (min_rest_ms, min_move_ms, sd_mult, speed_floor) in tqdm(
        grid, desc="Sweeping params"):
        total = 0
        edge_total = 0
        sess_with = 0
        for sid, (a, b) in sess_slices.items():
            s_arr = spd[a:b]
            c, e = _count_session(s_arr, min_rest_ms, min_move_ms, sd_mult, speed_floor)
            total += c
            edge_total += e
            if c > 0:
                sess_with += 1

        row = {
            'min_rest_ms': int(min_rest_ms),
            'min_move_ms': int(min_move_ms),
            'sd_mult': float(sd_mult),
            'speed_floor': float(speed_floor),
            'N_onsets': int(total),
            'sessions_with_onsets': int(sess_with),
        }
        if edge_window_s is not None:
            row['N_edge_safe'] = int(edge_total)
        rows.append(row)

    out = pd.DataFrame(rows).sort_values(
        ['N_edge_safe' if edge_window_s is not None else 'N_onsets',
         'sessions_with_onsets', 'N_onsets'],
        ascending=[False, False, False]
    ).reset_index(drop=True)
    return out
def materialize_initiation_simple(
    df,
    fps,
    min_rest_ms,
    min_move_ms,
    sd_mult,
    speed_floor,
    session_idx_col='session_idx',
    frame_col='frame_index',
    speed_col='velocity_px_s',
    omega_col='angular_velocity'
):
    # uses your existing make_initiation_series (speed + omega sign)
    init_label, init_onset = make_initiation_series(
        df,
        session_idx_col=session_idx_col,
        frame_col=frame_col,
        speed_col=speed_col,
        omega_col=omega_col,
        fps=fps,
        min_rest_ms=min_rest_ms,
        min_move_ms=min_move_ms,
        sd_mult=sd_mult,
        speed_floor=speed_floor
    )
    return init_label, init_onset