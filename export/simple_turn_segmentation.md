# Loading of sessions

## Imports and session loading


```python
import sys
from pathlib import Path

# --- 1. SETUP PROJECT PATH ---
print("Setting up project path...")
project_root = Path.cwd()
if not (project_root / "fiberphotometry").exists():
    project_root = project_root.parent
sys.path.insert(0, str(project_root))
print(f"Project root added to path: {project_root}")
```

    Setting up project path...
    Project root added to path: /home/fsp585/Desktop/FiberphotometryCode



```python
import pickle

load_filename = 'big_run_frac_1_ld10_jn24_with_merged.pkl'

print(f"--- 🚀 Loading state from '{load_filename}' ---")

with open(load_filename, 'rb') as f:
    state = pickle.load(f)

# unpack what we care about
sessions = state['sessions']
merged_all = state.get('merged_all')

print(f"✅ Loaded. Sessions: {len(sessions)}")
print("🔎 merged_all:", 
      "present" if merged_all is not None else "missing")

if merged_all is not None:
    print(f"   shape = {merged_all.shape[0]:,} × {merged_all.shape[1]}")
```

    --- 🚀 Loading state from 'big_run_frac_1_ld10_jn24_with_merged.pkl' ---
    ✅ Loaded. Sessions: 16
    🔎 merged_all: present
       shape = 3,544,889 × 26



```python
def add_session_idx(merged_all, sessions, session_base_col='session_base', h5_attr='kpms_h5_path',
                    strict=True, use_fallback=True):
    """
    Add a 'session_idx' column to merged_all by matching session_base to the stem of sessions[i].kpms_h5_path.
    - session_base_col: column in merged_all containing the expected stem (basename without '.h5')
    - h5_attr: attribute on each session that holds the full .h5 path
    - strict: if True, assert that all rows were matched (bijective assumption)
    - use_fallback: if True, try a light substring fallback for any unmatched stems

    Returns: merged_all (same object) with 'session_idx' added.
    """
    # 1) Build stem -> index map from sessions
    stem_to_idx = {}
    dups = []
    for i, s in enumerate(sessions):
        h5 = getattr(s, h5_attr, '') or ''
        fname = h5.rsplit('/', 1)[-1]            # basename
        stem = fname[:-3] if fname.endswith('.h5') else fname
        if stem in stem_to_idx:
            dups.append(stem)
        stem_to_idx[stem] = i

    if dups:
        print(f"⚠️ Duplicate stems found (unexpected for bijection): {dups[:5]}{' ...' if len(dups) > 5 else ''}")

    # 2) Direct map
    merged_all['session_idx'] = merged_all[session_base_col].map(stem_to_idx)

    # 3) Optional fallback for any misses (substring containment in either direction)
    if use_fallback:
        missing = merged_all.loc[merged_all['session_idx'].isna(), session_base_col].unique().tolist()
        if missing:
            for base in missing:
                cands = [idx for stem, idx in stem_to_idx.items() if (base in stem) or (stem in base)]
                if len(cands) == 1:
                    stem_to_idx[base] = cands[0]
            merged_all['session_idx'] = merged_all['session_idx'].fillna(
                merged_all[session_base_col].map(stem_to_idx)
            )

    # 4) Sanity / strictness
    n_missing = int(merged_all['session_idx'].isna().sum())
    if n_missing and strict:
        examples = merged_all.loc[merged_all['session_idx'].isna(), session_base_col].unique()[:5]
        raise ValueError(f"{n_missing} rows unmatched. Examples: {list(examples)}")
    elif n_missing:
        print(f"⚠️ {n_missing} rows unmatched; leaving session_idx as NaN for those rows.")

    return merged_all

```


```python
merged_all = add_session_idx(merged_all, sessions)  # adds 'session_idx' in place and returns df
```


```python
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

```




    'Arena1234_CenterTopCamVid_2024-12-13T12_20_06_bottom_leftDLC_HrnetW32_dataset_fixDec16shuffle10_detector_020_snapshot_020_part1'




```python
# Cell 1 — basic sanity & columns

print("rows:", len(merged_all))
print("cols:", len(merged_all.columns))
print("has session_idx:", 'session_idx' in merged_all.columns)

need = ['session_idx','frame_index','angular_velocity','velocity_px_s']
missing = [c for c in need if c not in merged_all.columns]
print("missing required cols:", missing)

print("nulls angular_velocity:", int(merged_all['angular_velocity'].isna().sum()))
print("nulls velocity_px_s:", int(merged_all['velocity_px_s'].isna().sum()))

print("example columns:", merged_all.columns[:10].tolist())

```

    rows: 3544889
    cols: 27
    has session_idx: True
    missing required cols: []
    nulls angular_velocity: 0
    nulls velocity_px_s: 0
    example columns: ['session_base', 'name', 'group', 'syllable', 'onset', 'frame_index', 'in_video_index', 'had_signal_match', 'centroid_x', 'centroid_y']



```python
# Cell 2 — per-session size and frame extents (helps spot weird resets)

gb = merged_all.groupby('session_idx')['frame_index']
sess_summary = gb.agg(['count','min','max']).reset_index()
sess_summary['frames_span'] = sess_summary['max'] - sess_summary['min'] + 1
print(sess_summary.head(10))
print("--- sessions:", len(sess_summary), 
      " median rows/session:", int(sess_summary['count'].median()))

```

       session_idx   count  min     max  frames_span
    0            0  231374    0  231373       231374
    1            1  220475    0   99313        99314
    2            2  218463    0  218462       218463
    3            3  222797    0  222796       222797
    4            4  196653    0  131867       131868
    5            5  221549    0  221548       221549
    6            6  221848    0  221847       221848
    7            7  222645    0  180141       180142
    8            8  253732    0  219607       219608
    9            9  206811    0  206810       206811
    --- sessions: 16  median rows/session: 221293



```python
# Cell 3 — global ω (angular_velocity) distribution (abs), quick percentiles

def percentiles(series, ps=(0,1,5,10,25,50,75,90,95,97.5,99,99.5,99.9,100)):
    vals = series.dropna().values
    vals.sort()
    out = {}
    n = len(vals)
    for p in ps:
        idx = int(round((p/100.0)*(n-1))) if n else 0
        out[p] = vals[idx] if n else None
    return out

omega_abs = merged_all['angular_velocity'].abs()
omega_pct = percentiles(omega_abs)
print("abs(ω) percentiles:", omega_pct)

```

    abs(ω) percentiles: {0: 0.0, 1: 0.0026429634692393122, 5: 0.013400013245552511, 10: 0.027120764200156966, 25: 0.07467071137299608, 50: 0.19752821986098623, 75: 0.445634502447827, 90: 0.811544817723896, 95: 1.092942276278486, 97.5: 1.3790932901948172, 99: 1.788285304328232, 99.5: 2.212323831457106, 99.9: 187.93391025749182, 100: 188.48944535319373}



```python
# Cell 4 — ω per-session percentiles (to see cross-session variability)

def per_session_percentiles(df, col, ps=(50,85,90,95,99)):
    rows = []
    for sid, sub in df.groupby('session_idx', sort=False):
        p = percentiles(sub[col].abs(), ps)
        row = {'session_idx': sid}
        for k,v in p.items(): row[f'p{k}'] = v
        rows.append(row)
    return rows

omega_by_session = per_session_percentiles(merged_all, 'angular_velocity')
print("example per-session ω percentiles (first 5):")
for r in omega_by_session[:5]:
    print(r)

```

    example per-session ω percentiles (first 5):
    {'session_idx': 1, 'p50': 0.10716637046018995, 'p85': 0.4001425512590895, 'p90': 0.5237696671504597, 'p95': 0.7439725410353759, 'p99': 1.3394166715337619}
    {'session_idx': 2, 'p50': 0.23989595561822893, 'p85': 0.7793771635734403, 'p90': 0.9691776653418782, 'p95': 1.2684689490900478, 'p99': 1.9298136051176984}
    {'session_idx': 0, 'p50': 0.2130983971678868, 'p85': 0.6563082578680257, 'p90': 0.8178190728057169, 'p95': 1.1046355246173167, 'p99': 1.8113083544240816}
    {'session_idx': 4, 'p50': 0.1659160201464005, 'p85': 0.6483581292567986, 'p90': 0.8044648480150585, 'p95': 1.0601680031960115, 'p99': 1.6430436845428011}
    {'session_idx': 3, 'p50': 0.2333063262866375, 'p85': 0.6992572075252346, 'p90': 0.8703298766784373, 'p95': 1.1674592335728073, 'p99': 1.8501148893557184}



```python
# Cell 5 — speed distribution & a candidate speed_floor (for curvature)

speed = merged_all['velocity_px_s'].clip(lower=0)  # ensure non-negative
speed_pct = percentiles(speed, ps=(0,1,5,10,20,30,50,70,80,90,95,99))
print("speed percentiles:", speed_pct)

# heuristic: take speed_floor near P20–P30 (below that often 'in-place' or jitter)
speed_floor_candidate = speed_pct[20] if speed_pct[20] is not None else 0.0
print("candidate speed_floor:", speed_floor_candidate)

```

    speed percentiles: {0: 0.0, 1: 2.56044857688789, 5: 5.831794935660727, 10: 8.411043522454428, 20: 12.424419256274003, 30: 16.003264594271947, 50: 23.6155878003493, 70: 35.37597317315197, 80: 47.03625754592847, 90: 68.5009393121939, 95: 86.20494964351823, 99: 116.83185202440298}
    candidate speed_floor: 12.424419256274003



```python
# Cell 6 — curvature κ = |ω| / v (fallback to |ω| when v < speed_floor)

fps = 80.0
eps = 1e-6
speed_floor = speed_floor_candidate  # tweak later

v = speed.values
w = merged_all['angular_velocity'].values
absw = [abs(x) for x in w]
kappa = [ (aw / (vv if vv > eps else eps)) if vv >= speed_floor else aw 
          for aw, vv in zip(absw, v) ]

# summary percentiles for κ
import numpy as np  # If you already have numpy; if not, you can skip np.array
# (you said you likely already have imports; otherwise just compute via dict below)
kappa_series = merged_all['angular_velocity'].copy()
kappa_series[:] = kappa
kappa_pct = percentiles(kappa_series)
print("κ percentiles:", kappa_pct)

```

    κ percentiles: {0: 0.0, 1: 0.00011012257950561201, 5: 0.000552855584623315, 10: 0.0011283814884661739, 25: 0.003214148678750121, 50: 0.009751569884824441, 75: 0.030438998264472786, 90: 0.14809514723988393, 95: 0.3458837450668617, 97.5: 0.5738066823554977, 99: 0.9131083552438968, 99.5: 1.2276607151610075, 99.9: 6.213160886979475, 100: 188.48576494729915}



```python
import pandas as pd
# Cell 7 — quick ω vs speed coupling check (correlation & top-tail)

# crude correlation (Pearson-ish without imports)
x = omega_abs.dropna().values
y = speed.loc[omega_abs.index].dropna().values
n = min(len(x), len(y))
x = x[:n]; y = y[:n]
mx = sum(x)/n; my = sum(y)/n
sx = (sum((xi-mx)**2 for xi in x)/max(n-1,1))**0.5
sy = (sum((yi-my)**2 for yi in y)/max(n-1,1))**0.5
cov = sum((xi-mx)*(yi-my) for xi,yi in zip(x,y))/max(n-1,1)
corr = cov / (sx*sy + 1e-12)
print("corr(abs(ω), speed):", round(corr, 4))

# top tail overlap: how many frames in top 1% of ω are also top 1% speed?
def top_mask(vals, pct=99):
    thr = percentiles(pd.Series(vals), ps=(pct,)) [pct]
    return np.array(vals) >= thr if 'np' in globals() else [v>=thr for v in vals]

mask_w = top_mask(omega_abs, 99)
mask_v = top_mask(speed, 99)
overlap = (mask_w & mask_v).sum() if 'np' in globals() else sum(1 for a,b in zip(mask_w,mask_v) if a and b)
print("overlap of top 1% |ω| and top 1% speed:", int(overlap))

```

    corr(abs(ω), speed): 0.0176
    overlap of top 1% |ω| and top 1% speed: 553



```python
# Cell 8 — angular acceleration α ~ Δω/Δt distribution (helps pick clip_hi_pct, smooth_win)

# approximate α in units of (ω units)/s via finite diff * fps
w_values = merged_all['angular_velocity'].values
# we must reset between sessions to avoid cross-session spikes
alpha = [0.0]*len(merged_all)
idx = merged_all.sort_values(['session_idx','frame_index']).index.to_list()
sess = merged_all['session_idx'].values

for k in range(1, len(idx)):
    i = idx[k]; j = idx[k-1]
    if sess[i] != sess[j]:
        alpha[i] = 0.0
    else:
        alpha[i] = (w_values[i] - w_values[j]) * fps  # Δω * fps

alpha_series = merged_all['angular_velocity'].copy()
alpha_series[:] = alpha
alpha_pct = percentiles(alpha_series.abs(), ps=(50,90,95,99,99.5,99.9,100))
print("abs(α) percentiles:", alpha_pct)

# suggestion: pick clip_hi_pct so that abs(ω) 99.5–99.9 trims crazy α spikes

```

    abs(α) percentiles: {50: 2.7594835038657006, 90: 13.608605256636963, 95: 33.08502599918093, 99: 99.27107447235261, 99.5: 171.10009842442508, 99.9: 15082.379974111393, 100: 30157.686644082405}



```python
# Cell 9 — smoothing benefit test (variance & top-tail reduction across windows)

def moving_avg_list(x, k):
    if k <= 1: return list(x)
    pad = k//2
    seq = [x[0]]*pad + list(x) + [x[-1]]*pad
    out = []
    for i in range(len(x)):
        out.append(sum(seq[i:i+k]) / float(k))
    return out

windows = [1,3,5,7,9]
omega_by_win = {}
ordered_idx = merged_all.sort_values(['session_idx','frame_index']).index.to_list()
w_ordered = merged_all.loc[ordered_idx, 'angular_velocity'].values

for k in windows:
    sm = moving_avg_list(w_ordered, k)
    # compute 99th percentile of abs after smoothing
    s = pd.Series(sm).abs()
    p = percentiles(s, ps=(95,99,99.5))
    omega_by_win[k] = p

print("abs(ω) tail percentiles by smooth window:", omega_by_win)
# look for a window (e.g., 5–9) that meaningfully reduces 99–99.5th without flattening too much

```

    abs(ω) tail percentiles by smooth window: {1: {95: 1.092942276278486, 99: 1.788285304328232, 99.5: 2.212323831457106}, 3: {95: 1.095016048150299, 99: 2.0216739511286153, 99.5: 61.58148487388698}, 5: {95: 1.114585403259201, 99: 3.867182953298088, 99.5: 37.1641431041346}, 7: {95: 1.1330975531393577, 99: 26.01895249595359, 99.5: 26.554741777197005}, 9: {95: 1.1498253350541414, 99: 20.325248641359348, 99.5: 20.647259460940962}}



```python
# Cell 10 — candidate start percentiles for bout detection (ω and κ)

cands = (85, 90, 95)
stats = {'omega': {}, 'kappa': {}}
for p in cands:
    stats['omega'][p] = percentiles(omega_abs, ps=(p,))[p]
    stats['kappa'][p] = percentiles(kappa_series, ps=(p,))[p]
print("start thresholds (global):", stats)

# per-session for the chosen p (e.g., 90)
p = 90
per_sess_thr = []
for sid, sub in merged_all.groupby('session_idx', sort=False):
    per_sess_thr.append({
        'session_idx': sid,
        'omega_start_90': percentiles(sub['angular_velocity'].abs(), ps=(p,))[p],
        'kappa_start_90': percentiles(pd.Series(kappa_series.loc[sub.index]), ps=(p,))[p],
    })
print("example per-session starts (first 5):", per_sess_thr[:5])

```

    start thresholds (global): {'omega': {85: 0.6479088841706804, 90: 0.811544817723896, 95: 1.092942276278486}, 'kappa': {85: 0.06889015000457244, 90: 0.14809514723988393, 95: 0.3458837450668617}}
    example per-session starts (first 5): [{'session_idx': 1, 'omega_start_90': 0.5237696671504597, 'kappa_start_90': 0.1444267994648274}, {'session_idx': 2, 'omega_start_90': 0.9691776653418782, 'kappa_start_90': 0.09132087523094246}, {'session_idx': 0, 'omega_start_90': 0.8178190728057169, 'kappa_start_90': 0.1855419025644811}, {'session_idx': 4, 'omega_start_90': 0.8044648480150585, 'kappa_start_90': 0.10431842456619655}, {'session_idx': 3, 'omega_start_90': 0.8703298766784373, 'kappa_start_90': 0.1819292129514105}]



```python
# Cell 11 — above-threshold run-lengths (no hysteresis yet), to size min_bout_ms

def run_lengths(mask):
    lengths = []
    i = 0; n = len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j]:
            j += 1
        lengths.append(j - i)
        i = j
    return lengths

# pick global ω start (e.g., p90) and make mask per session to avoid boundary issues
p = 90
omega_thr = stats['omega'][p]
lengths = []
for sid, sub in merged_all.groupby('session_idx', sort=False):
    mask = (sub['angular_velocity'].abs().values >= omega_thr)
    lengths.extend(run_lengths(mask))

# summarize run lengths in frames and ms
def summarize_lengths(frames_list, fps=80.0):
    if not frames_list:
        return {}
    s = sorted(frames_list)
    res = {}
    for p in (50,75,90,95,99):
        idx = int(round((p/100.0)*(len(s)-1)))
        res[f'len_p{p}_frames'] = s[idx]
        res[f'len_p{p}_ms'] = 1000.0 * s[idx] / fps
    return res

len_stats = summarize_lengths(lengths, fps=fps)
print("ω>=p90 run-length stats:", len_stats)
# choose min_bout_ms slightly below len_p50_ms, or use p25 if you compute it similarly

```

    ω>=p90 run-length stats: {'len_p50_frames': 8, 'len_p50_ms': 100.0, 'len_p75_frames': 16, 'len_p75_ms': 200.0, 'len_p90_frames': 26, 'len_p90_ms': 325.0, 'len_p95_frames': 32, 'len_p95_ms': 400.0, 'len_p99_frames': 44, 'len_p99_ms': 550.0}



```python
# Cell 12 — gap lengths between above-threshold runs (to size merge_gap_ms)

def gap_lengths(mask):
    gaps = []
    n = len(mask)
    # find runs first
    runs = []
    i=0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j]:
            j += 1
        runs.append((i,j-1))
        i = j
    # now compute gaps between runs
    for a,b in zip(runs, runs[1:]):
        gap = b[0] - a[1] - 1
        gaps.append(gap)
    return gaps

p = 90
omega_thr = stats['omega'][p]
gaps_all = []
for sid, sub in merged_all.groupby('session_idx', sort=False):
    mask = (sub['angular_velocity'].abs().values >= omega_thr)
    gaps_all.extend(gap_lengths(mask))

gap_stats = summarize_lengths(gaps_all, fps=fps)
print("inter-run gap stats (frames/ms) for ω>=p90:", gap_stats)
# choose merge_gap_ms around p50–p75 of these tiny gaps to fuse flickers

```

    inter-run gap stats (frames/ms) for ω>=p90: {'len_p50_frames': 44, 'len_p50_ms': 550.0, 'len_p75_frames': 110, 'len_p75_ms': 1375.0, 'len_p90_frames': 228, 'len_p90_ms': 2850.0, 'len_p95_frames': 352, 'len_p95_ms': 4400.0, 'len_p99_frames': 858, 'len_p99_ms': 10725.0}



```python
# Cell 13 — left/right balance at top turns (peak frames in ω top-tail)

# use ω top 1% frames, count sign
thr = percentiles(omega_abs, ps=(99,))[99]
top = merged_all.loc[omega_abs >= thr, 'angular_velocity']
lr_counts = {'left': int((top < 0).sum()), 'right': int((top >= 0).sum())}
print("L/R counts in top 1% |ω| frames:", lr_counts)

```

    L/R counts in top 1% |ω| frames: {'left': 20025, 'right': 15425}



```python
# Cell 14 — kappa run-lengths and gaps (repeat of 11–12 for κ), helps if you switch method

p = 90
k_thr = stats['kappa'][p]
lengths_k = []
gaps_k = []
for sid, sub in merged_all.groupby('session_idx', sort=False):
    m = (kappa_series.loc[sub.index].values >= k_thr)
    lengths_k.extend(run_lengths(m))
    gaps_k.extend(gap_lengths(m))

len_k_stats = summarize_lengths(lengths_k, fps=fps)
gap_k_stats = summarize_lengths(gaps_k, fps=fps)
print("κ>=p90 run-length stats:", len_k_stats)
print("κ>=p90 inter-run gaps:", gap_k_stats)

```

    κ>=p90 run-length stats: {'len_p50_frames': 1, 'len_p50_ms': 12.5, 'len_p75_frames': 1, 'len_p75_ms': 12.5, 'len_p90_frames': 2, 'len_p90_ms': 25.0, 'len_p95_frames': 3, 'len_p95_ms': 37.5, 'len_p99_frames': 4, 'len_p99_ms': 50.0}
    κ>=p90 inter-run gaps: {'len_p50_frames': 4, 'len_p50_ms': 50.0, 'len_p75_frames': 10, 'len_p75_ms': 125.0, 'len_p90_frames': 29, 'len_p90_ms': 362.5, 'len_p95_frames': 54, 'len_p95_ms': 675.0, 'len_p99_frames': 119, 'len_p99_ms': 1487.5}



```python
# Cell 15 — draft recommendations based on the above quick stats

reco = {}

# speed_floor: from Cell 5
reco['speed_floor'] = float(speed_floor_candidate or 0.0)

# smooth_win: from Cell 9 tail suppression — pick smallest k that lowers p99 notably
# (Here, we just propose 5 as a sane default; adjust after reading Cell 9)
reco['smooth_win'] = 5

# clip_hi_pct: from Cell 8 (α tails) — start with 99.5 to trim spikes
reco['clip_hi_pct'] = 99.5

# start percentile (ω): from Cell 10 — start at 90; if too many events, try 95
reco['turn_start_pct'] = 90
reco['stop_ratio'] = 0.6

# min_bout_ms: from Cell 11 len stats — pick around p50 frames if you want stability
if 'len_p50_ms' in len_stats and len_stats['len_p50_ms'] is not None:
    reco['min_bout_ms'] = max(80, int(round(len_stats['len_p50_ms'])))  # at least one frame chunk
else:
    reco['min_bout_ms'] = 120

# merge_gap_ms: from Cell 12 gap stats — p50–p75
mg = None
for key in ('len_p50_ms','len_p75_ms'):
    if key in gap_stats and gap_stats[key] is not None:
        mg = int(round(gap_stats[key])); break
reco['merge_gap_ms'] = mg if mg is not None else 120

print("🔧 parameter recommendations (draft):")
for k,v in reco.items():
    print(f"  {k}: {v}")

```

    🔧 parameter recommendations (draft):
      speed_floor: 12.424419256274003
      smooth_win: 5
      clip_hi_pct: 99.5
      turn_start_pct: 90
      stop_ratio: 0.6
      min_bout_ms: 100
      merge_gap_ms: 550



```python
# Cell 16 — (optional) per-session recommendations (ω p90 and p95 starts)

per_sess_reco = []
for r in omega_by_session:
    per_sess_reco.append({
        'session_idx': r['session_idx'],
        'omega_start_85': r.get('p85'),
        'omega_start_90': r.get('p90'),
        'omega_start_95': r.get('p95'),
        'omega_start_99': r.get('p99'),
    })
print("per-session start thresholds (first 10):")
for row in per_sess_reco[:10]:
    print(row)

```

    per-session start thresholds (first 10):
    {'session_idx': 1, 'omega_start_85': 0.4001425512590895, 'omega_start_90': 0.5237696671504597, 'omega_start_95': 0.7439725410353759, 'omega_start_99': 1.3394166715337619}
    {'session_idx': 2, 'omega_start_85': 0.7793771635734403, 'omega_start_90': 0.9691776653418782, 'omega_start_95': 1.2684689490900478, 'omega_start_99': 1.9298136051176984}
    {'session_idx': 0, 'omega_start_85': 0.6563082578680257, 'omega_start_90': 0.8178190728057169, 'omega_start_95': 1.1046355246173167, 'omega_start_99': 1.8113083544240816}
    {'session_idx': 4, 'omega_start_85': 0.6483581292567986, 'omega_start_90': 0.8044648480150585, 'omega_start_95': 1.0601680031960115, 'omega_start_99': 1.6430436845428011}
    {'session_idx': 3, 'omega_start_85': 0.6992572075252346, 'omega_start_90': 0.8703298766784373, 'omega_start_95': 1.1674592335728073, 'omega_start_99': 1.8501148893557184}
    {'session_idx': 6, 'omega_start_85': 0.5813648589851672, 'omega_start_90': 0.7183498714260774, 'omega_start_95': 0.9620934517221913, 'omega_start_99': 1.6304205266344818}
    {'session_idx': 7, 'omega_start_85': 0.44544188006681473, 'omega_start_90': 0.5716033265769482, 'omega_start_95': 0.7984432552337228, 'omega_start_99': 1.4054153205258824}
    {'session_idx': 5, 'omega_start_85': 0.8199360439616044, 'omega_start_90': 1.010125674782889, 'omega_start_95': 1.3111310015206534, 'omega_start_99': 1.9408774782938054}
    {'session_idx': 9, 'omega_start_85': 0.6019430212721477, 'omega_start_90': 0.7327792083495266, 'omega_start_95': 0.9441297739070698, 'omega_start_99': 1.4470944945358566}
    {'session_idx': 10, 'omega_start_85': 0.43063803583631044, 'omega_start_90': 0.5407813439292375, 'omega_start_95': 0.7260269803801167, 'omega_start_99': 1.1266866190212488}



```python
# Cell 17 — (optional) fraction of frames classified as candidate turn at p90 (ω and κ)

def frac_above(series, thr):
    n = series.notna().sum()
    if n == 0: return 0.0
    return float((series >= thr).sum()) / n

omega_thr90 = stats['omega'][90]
kappa_thr90 = stats['kappa'][90]

frac_omega = frac_above(omega_abs, omega_thr90)
frac_kappa = frac_above(kappa_series, kappa_thr90)
print(f"fraction frames ω >= p90: {frac_omega:.3f}")
print(f"fraction frames κ >= p90: {frac_kappa:.3f}")

```

    fraction frames ω >= p90: 0.100
    fraction frames κ >= p90: 0.100



```python
# Cell 18 — (optional) check for session boundary artifacts: large |Δω| at boundary

# Count how many of the top 0.1% |α| entries occur on the first frame of a session
alpha_abs = alpha_series.abs()
thr_a = percentiles(alpha_abs, ps=(99.9,))[99.9]
is_boundary = pd.Series(False, index=merged_all.index)
# mark first index of each session (in frame order)
first_idx = merged_all.sort_values(['session_idx','frame_index']).groupby('session_idx').head(1).index
is_boundary.loc[first_idx] = True

boundary_hits = int(((alpha_abs >= thr_a) & is_boundary).sum())
print("top 0.1% |α| that are session-boundary frames:", boundary_hits)

```

    top 0.1% |α| that are session-boundary frames: 0



```python

```
