from .config import PROJECT_DIR_FULL
from .detect import make_turn_syllable_series
from .utils import _ensure_dir


def _param_hash(d: dict) -> str:
    import hashlib
    import json
    whitelist = [
        'method','fps','turn_start_pct','stop_ratio','min_bout_ms','merge_gap_ms',
        'smooth_win','clip_hi_pct','speed_floor',
        'session_idx_col','frame_col','omega_col','speed_col'
    ]
    dd = {k: d.get(k) for k in whitelist}
    s = json.dumps(dd, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(s.encode('utf-8')).hexdigest()[:10]

def build_or_load_turn_index(
    df, dataset_tag: str, detector_kwargs: dict,
    cache_dir=None
):
    """
    Run detection (non-percentile params), extract per-bout rows with peak |omega| & |kappa|,
    attach argsort ranks for fast binning, and cache to disk.
    """
    import os
    import time

    import numpy as np
    import pandas as pd

    cache_dir = cache_dir or os.path.join((PROJECT_DIR_FULL or '.'), 'turn_cache')
    _ensure_dir(cache_dir)

    # identity key
    h = _param_hash(detector_kwargs)
    fname = f"turnindex__{dataset_tag}__{h}.pkl"
    fpath = os.path.join(cache_dir, fname)

    # try load
    if os.path.isfile(fpath):
        obj = pd.read_pickle(fpath)
        # sanity minimal schema
        required_cols = {'onset_df_idx','session_idx','start_i','end_i','dir',
                         'peak_abs_omega','peak_abs_kappa','rank_omega','rank_kappa'}
        if required_cols.issubset(set(obj['turn_index'].columns)):
            print(f"📦 Loaded turn index cache: {fpath}")
            return obj
        else:
            print(f"⚠️ Cache schema changed; rebuilding: {fpath}")

    # --- build fresh ---
    # 1) run detector to get per-frame labels & onsets
    turn_syll, turn_on = make_turn_syllable_series(df, **detector_kwargs)

    # 2) ensure sorted & aligned
    work = df.sort_values(['session_idx','frame_index']).reset_index()
    # keep original df index to map back later
    work.rename(columns={'index':'orig_index'}, inplace=True)
    # align detector outputs to this sorted view
    # (assumes your detector used the same ordering or returned Series aligned to df index)
    if getattr(turn_on, 'index', None) is not None and not turn_on.index.equals(df.index):
        # align by original index
        turn_on = turn_on.reindex(df.index)
        turn_syll = turn_syll.reindex(df.index)

    turn_on_sorted   = turn_on.loc[work['orig_index']].to_numpy()
    turn_syll_sorted = turn_syll.loc[work['orig_index']].astype('object').to_numpy()

    sess  = work['session_idx'].to_numpy()
    omega = np.abs(work['angular_velocity'].to_numpy(dtype='float64'))
    speed = work['velocity_px_s'].to_numpy(dtype='float64')
    # curvature κ ≈ dθ/ds = ω / speed (guard with speed_floor)
    speed_floor = float(detector_kwargs.get('speed_floor', 2.0 if detector_kwargs.get('method','omega')=='omega' else 10.0))
    denom = np.maximum(speed, speed_floor)
    kappa = np.abs(omega / denom)

    n = len(work)
    rows = []
    i = 0
    while i < n:
        if not turn_on_sorted[i]:
            i += 1
            continue
        # start of bout
        s_i = i
        lab = turn_syll_sorted[i]
        # robust direction parse
        d = None
        if isinstance(lab, str):
            lo = lab.lower()
            if lo.startswith('left'):  d = 'left'
            elif lo.startswith('right'): d = 'right'
        # walk until label changes or session boundary
        j = i + 1
        while j < n and (sess[j] == sess[i]) and (turn_syll_sorted[j] == lab):
            j += 1
        e_i = j  # exclusive end

        # peaks within bout [s_i:e_i)
        peak_w = float(np.nanmax(omega[s_i:e_i])) if e_i > s_i else float('nan')
        peak_k = float(np.nanmax(kappa[s_i:e_i])) if e_i > s_i else float('nan')

        rows.append({
            'onset_df_idx': int(work.at[s_i, 'orig_index']),  # index in the original df
            'onset_sorted_i': int(s_i),                       # position in sorted view
            'session_idx': int(sess[s_i]),
            'start_i': int(s_i),
            'end_i': int(e_i),
            'dir': d,  # 'left'/'right'/None
            'label': lab if isinstance(lab, str) else None,
            'peak_abs_omega': peak_w,
            'peak_abs_kappa': peak_k,
        })
        i = j

    turn_index = pd.DataFrame(rows)
    if turn_index.empty:
        raise RuntimeError("No bouts detected; check detector params or inputs.")

    # 3) attach argsort ranks (ascending; higher ranks = stronger turns)
    order_w = np.argsort(turn_index['peak_abs_omega'].to_numpy())
    order_k = np.argsort(turn_index['peak_abs_kappa'].to_numpy())

    rank_w = np.empty(len(turn_index), dtype='int32'); rank_w[order_w] = np.arange(len(turn_index), dtype='int32')
    rank_k = np.empty(len(turn_index), dtype='int32'); rank_k[order_k] = np.arange(len(turn_index), dtype='int32')

    turn_index['rank_omega'] = rank_w
    turn_index['rank_kappa'] = rank_k

    meta = {
        'dataset_tag': dataset_tag,
        'detector_hash': h,
        'detector_kwargs': {k: detector_kwargs.get(k) for k in detector_kwargs},
        'n_bouts': int(len(turn_index)),
        'n_frames': int(len(df)),
        'version': 1,
        'time': time.strftime('%Y-%m-%d %H:%M:%S')
    }

    out = {'turn_index': turn_index, 'meta': meta}
    pd.to_pickle(out, fpath)
    print(f"💾 Built & saved turn index: {fpath}  | bouts={len(turn_index)}")
    return out

def slice_by_percentile(turn_index, metric='omega', p_lo=80, p_hi=85):
    """
    Return a view (DataFrame) of bouts whose rank is within [p_lo, p_hi) *by rank*.
    """
    import numpy as np
    mcol = 'rank_omega' if metric == 'omega' else 'rank_kappa'
    N = len(turn_index)
    lo = int(np.floor(p_lo/100.0 * N))
    hi = int(np.floor(p_hi/100.0 * N))  # exclusive
    lo = max(0, min(N, lo))
    hi = max(lo, min(N, hi))
    return turn_index.nsmallest(hi, mcol).iloc[lo:hi]

def onset_mask_from_slice(df, slice_df, onset_col='turn_onset_cache'):
    """
    Build a boolean mask Series aligned to df.index with True at selected onsets.
    """
    import pandas as pd
    mask = pd.Series(False, index=df.index)
    idxs = slice_df['onset_df_idx'].to_numpy()
    mask.loc[idxs] = True
    df[onset_col] = mask
    return onset_col
