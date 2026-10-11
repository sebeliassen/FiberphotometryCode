import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt


def list_combined_csv_entries(root, jcfg):
    import re
    from pathlib import Path

    from .time import parse_csv_ts

    csv_re = re.compile(jcfg.phot_combined_csv_regex, re.I)
    entries = []
    for p in Path(root).rglob(jcfg.phot_combined_csv_glob):
        m = csv_re.match(p.name)
        if not m:
            continue
        try:
            entries.append((p, parse_csv_ts(m.group(1)), p.parent.name))
        except Exception:
            pass
    entries.sort(key=lambda t: t[1])
    return entries

def best_csv_match(dt_session, entries, tol_seconds):
    # pick csv with min |Δt| within tolerance
    best = None
    for p, dt_csv, trial_id in entries:
        delta = abs((dt_session - dt_csv).total_seconds())
        if delta <= tol_seconds and (best is None or delta < best[0]):
            best = (delta, p, dt_csv, trial_id)
    return None if best is None else best[1:]  # (path, dt_csv, trial_id)

def plan_from_header(header_cols, session, jcfg):
    import re
    pat = re.compile(jcfg.fiber_col_regex)
    led_col = getattr(jcfg, "led_col_name", "LedState")

    non_fiber_cols, fiber_by_idx = [], {}
    for c in header_cols:
        m = pat.match(str(c))
        if m:
            idx = int(m.group(2))
            fiber_by_idx.setdefault(idx, c)
        else:
            non_fiber_cols.append(c)

    eff = session.metadata.get("fiber_indices_effective") or session.metadata.get("fiber_indices_auto")
    if not isinstance(eff, (list, tuple)) or len(eff) == 0:
        ch_idx = session.metadata.get("chamber_index")
        if ch_idx is None:
            raise ValueError(f"{session.session_id}: missing fiber indices and chamber_index; set metadata['fiber_indices_effective'] or provide chamber_index.")
        eff = jcfg.default_effective_indices(int(ch_idx))
        session.metadata["fiber_indices_auto"] = eff
        session.metadata.setdefault("fiber_select_mode", jcfg.fiber_auto_pair_name)
    eff = [int(x) for x in eff]

    signals = []
    for k, idx in enumerate(eff):
        col = fiber_by_idx.get(idx)
        if col is not None:
            signals.append((f"signal_{k}", col))

    return {"non_fiber_cols": non_fiber_cols, "signals": signals, "effective_indices": eff, "led_col": led_col}

def update_signal_meta_from_signals(session, signals):
    _DASHY = r"[\s_\-\u2010\u2011\u2012\u2013\u2014\u2015\u2212]+"
    def _norm_label(s):
        if not isinstance(s, str) or not s: return None
        return re.sub(_DASHY, "-", s).upper()
    def _split_hemi_region(lbl):
        if not lbl: return (None, None)
        parts = lbl.split("-", 1)
        return (parts[0], parts[1]) if len(parts) == 2 else (None, lbl)

    pat = re.compile(r"^([A-Z]+)(\d+)$")
    sm = session.metadata.setdefault("signal_meta", {})
    fiber_to_signal = {}

    for sig_key, col in signals:
        e = sm.get(sig_key, {})
        e["orig_fiber"] = col
        m = pat.match(str(col))
        if m:
            e["orig_letter"] = m.group(1).upper()
            e["fiber_num"] = m.group(2)

        seed_label = session.metadata.get(sig_key)  # optional metadata.signal_0/_1 seed
        if seed_label:
            lbl = _norm_label(seed_label)
            hemi, region = _split_hemi_region(lbl)
            if hemi:   e.setdefault("hemisphere", hemi)
            if region: e.setdefault("brain_reg", region)

        sm[sig_key] = e
        fiber_to_signal[col] = sig_key

    session.metadata["signal_meta"] = sm
    return fiber_to_signal

def dedup_preserve_order(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            out.append(x); seen.add(x)
    return out

def select_signals_for_state(letter_key, signal_keys, sig_meta):
    # ISO keeps all signals; others filter by orig_letter
    if str(letter_key).lower() == "iso":
        return set(signal_keys)
    want = str(letter_key).upper()
    return {
        sk for sk in signal_keys
        if (sig_meta.get(sk, {}) or {}).get("orig_letter", "").upper() == want
    }

def get_signal_cols(df, jcfg=None):
    pat = re.compile((jcfg.signal_col_regex if jcfg else r"^signal_\d+$"))
    return [c for c in df.columns if pat.match(str(c))]

def compute_dff_and_z(iso_series, tgt_series):
    iso = iso_series.replace(0, pd.NA)
    dff = (tgt_series - iso) / iso
    mu = dff.mean(skipna=True)
    sd = dff.std(skipna=True)
    if sd is None or pd.isna(sd) or sd == 0:
        z = dff * 0  # all zeros
    else:
        z = (dff - mu) / sd
    return dff, z


def parse_trial_tokens(trial_id: str) -> List[str]:
    """
    'T1_1658.740.1666.e' -> ['1658','740','1666','e']
    'T2_7.8.3.2.4.e.6'   -> ['7','8','3','2','4','e','6']
    """
    if not trial_id:
        return []
    pos = trial_id.find("_")
    ids_part = trial_id[pos + 1 :] if pos >= 0 else trial_id
    return [t.strip() for t in ids_part.split(".") if t.strip() != ""]

def animal_id_for(trial_id: str, chamber_index: int) -> Optional[str]:
    toks = parse_trial_tokens(trial_id)
    if chamber_index >= len(toks):
        return None
    tok = toks[chamber_index]
    return None if tok.lower() == "e" else tok

def excel_col_label(idx: int) -> str:
    """0 -> 'A', 1 -> 'B', ..., 25 -> 'Z', 26 -> 'AA', ..."""
    if idx < 0:
        raise ValueError(f"Invalid chamber_index {idx}")
    s = ""
    n = idx
    while True:
        n, r = divmod(n, 26)
        s = chr(ord('A') + r) + s
        if n == 0:
            return s
        n -= 1  # Excel-style carry

_FIBER_SPLIT_RE = re.compile(r"[,\|\s;]+")

def parse_fibers_field(s: str) -> List[int]:
    # accepts "0,1", "0|1", "0 1", "0;1"
    parts = _FIBER_SPLIT_RE.split(str(s).strip())
    out = []
    for p in parts:
        if not p:
            continue
        try:
            out.append(int(p))
        except Exception:
            pass
    return out

def read_fiber_map(mapping_csv: Path) -> List[Dict]:
    """
    Expected columns:
      - trial_id (str), fibers (str like '0,1')
      - optional: chamber_index (int), animal_id (str)
    Returns list of dict rows.
    """
    rows: List[Dict] = []
    if not mapping_csv.exists():
        return rows
    mdf = pd.read_csv(mapping_csv)
    for _, row in mdf.iterrows():
        tr = str(row.get("trial_id", "")).strip()
        fb = row.get("fibers", "")
        if not tr or fb is None:
            continue
        fibers = parse_fibers_field(str(fb))
        ci = row.get("chamber_index", None)
        ai = row.get("animal_id", None)
        rows.append({
            "trial_id": tr,
            "chamber_index": None if pd.isna(ci) else int(ci),
            "animal_id": None if (ai is None or (isinstance(ai, float) and pd.isna(ai))) else str(ai),
            "fibers": fibers,
        })
    return rows

def select_fibers_for(
    trial_id: str,
    chamber_index: int,
    animal_id: Optional[str],
    manual_rows: List[Dict],
) -> Tuple[List[int], str]:
    """
    Return (fiber_indices, mode), preferring manual rows if available.
    Scoring: trial_id is pre-filter; then chamber_index match > animal_id match > trial-only.
    """
    if manual_rows:
        candidates = [r for r in manual_rows if r["trial_id"] == trial_id]
        if candidates:
            def score(r):
                s = 1  # trial match
                if r["chamber_index"] is not None and r["chamber_index"] == chamber_index: s += 4
                if r["animal_id"] is not None and animal_id is not None and r["animal_id"] == animal_id: s += 2
                return s
            best = max(candidates, key=score)
            if best.get("fibers"):
                return list(best["fibers"]), "manual"
    # fallback: auto pairing [2k, 2k+1]
    return [2 * chamber_index, 2 * chamber_index + 1], "auto_pair"

def filter_fiber_columns(
    df: pd.DataFrame,
    fiber_indices: List[int],
    non_fiber_cols: Tuple[str, ...],
    fiber_col_regex: str,
) -> pd.DataFrame:
    """
    Keep non-fiber cols + fiber columns whose trailing numeric index ∈ fiber_indices.
    """
    pat = re.compile(fiber_col_regex)
    keep = set(non_fiber_cols)
    for col in df.columns:
        m = pat.match(str(col))
        if m:
            idx = int(m.group(2))
            if idx in set(fiber_indices):
                keep.add(col)
    # preserve original column order
    cols = [c for c in df.columns if c in keep]
    return df[cols]

def list_alltimestamps_entries(dir_path: Path, jcfg):
    """
    Return [(path, dt, camera_name)] sorted by dt for files like:
    CenterTopCam_AllTimestampsCombined2024-12-13T13_37_45.csv
    """
    from .time import parse_session_ts
    csv_re = re.compile(jcfg.alltimestamps_csv_regex, re.I)
    out = []
    for p in Path(dir_path).glob(jcfg.alltimestamps_csv_glob):
        m = csv_re.match(p.name)
        if not m:
            continue
        try:
            dt = parse_session_ts(m.group("ts"))   # filename → datetime (only for file selection)
            cam = m.group("camera")
            out.append((p, dt, cam))
        except Exception:
            pass
    out.sort(key=lambda t: t[1])
    return out  # [(Path, datetime, camera_name)]

# attach nearest index from AllTimestampsCombined to photometry df
def attach_frame_idx_column(phot_df: pd.DataFrame, allts_df: pd.DataFrame, jcfg) -> pd.DataFrame:
    """
    Add jcfg.frame_index_col_name by nearest match between:
      phot_df[jcfg.phot_system_ts_col]  (int seconds)
      allts_df[jcfg.allts_system_ts_col] (int seconds)
    """
    pcol = jcfg.phot_system_ts_col
    acol = jcfg.allts_system_ts_col
    outcol = jcfg.frame_index_col_name

    if pcol not in phot_df.columns or acol not in allts_df.columns:
        return phot_df

    # numeric arrays (float for vector math; original are int seconds)
    p_ts = pd.to_numeric(phot_df[pcol], errors="coerce").to_numpy(dtype="float64")
    a_ts = pd.to_numeric(allts_df[acol], errors="coerce").to_numpy(dtype="float64")

    if a_ts.size == 0:
        phot_df = phot_df.copy()
        phot_df[outcol] = pd.Series([-1] * len(phot_df), dtype="Int64")
        return phot_df

    # ensure reference times are sorted for searchsorted
    order = np.argsort(a_ts, kind="mergesort")
    a_sorted = a_ts[order]

    # nearest-neighbor (left vs right)
    idx = np.searchsorted(a_sorted, p_ts, side="left")
    idx0 = np.clip(idx - 1, 0, len(a_sorted) - 1)
    idx1 = np.clip(idx,       0, len(a_sorted) - 1)

    left_dist  = np.abs(p_ts - a_sorted[idx0])
    right_dist = np.abs(a_sorted[idx1] - p_ts)
    pick_sorted = np.where(left_dist <= right_dist, idx0, idx1)

    # map back to original allts order; mask NaNs from p_ts
    mapped = order[pick_sorted]
    valid = ~np.isnan(p_ts)
    out_idx = np.full(p_ts.shape, -1, dtype=np.int64)
    out_idx[valid] = mapped[valid]

    phot_df = phot_df.copy()
    phot_df[outcol] = pd.Series(out_idx, dtype="Int64")
    return phot_df

def estimate_fps(df: pd.DataFrame, time_col: str = "SystemTimestamp") -> float:
    """Estimate sampling rate from timestamp column."""
    if time_col not in df.columns or len(df) < 2:
        return 30.0 # Fallback

    # Assuming seconds; robust diff
    diffs = np.diff(df[time_col])
    diffs = diffs[diffs > 0] # ignore resets/duplicates
    if len(diffs) == 0:
        return 30.0

    median_dt = np.nanmedian(diffs)
    if median_dt == 0:
        return 30.0
    return 1.0 / median_dt

def calculate_rolling_dff(series: pd.Series, fps: float, win_sec: float = 5.0, quantile: float = 0.1) -> pd.Series:
    """Datta-style dF/F0: F0 is rolling 10th percentile."""
    window = int(win_sec * fps)
    if window < 1: window = 1

    # Center=True is important for phase alignment
    rolling_f0 = series.rolling(window=window, center=True, min_periods=1).quantile(quantile)

    # Avoid division by zero
    rolling_f0 = rolling_f0.replace(0, np.nan)

    dff = (series - rolling_f0) / rolling_f0
    return dff


def calculate_ransac_subtraction(sig_dff: pd.Series, ref_dff: pd.Series) -> pd.Series:
    """
    Fit Ref to Sig using RANSAC and subtract.
    INCLUDES: 3Hz Low-Pass Filter on Reference (Datta Method).
    """
    # [cite_start]1. Filter Reference (3Hz Low Pass) [cite: 1142]
    # Assuming ~20-30Hz effective sampling rate.
    # We need to know FPS. If you don't pass FPS, assume 20Hz (standard for 80Hz/4-cycle).
    fps = 20.0
    b, a = butter(2, 3.0, fs=fps, btype='low')

    # Fill NaNs    # Updated to avoid FutureWarning
    ref_clean = ref_dff.ffill().bfill().fillna(0)
    ref_filtered_vals = filtfilt(b, a, ref_clean.values)
    ref_filtered = pd.Series(ref_filtered_vals, index=ref_dff.index)

    # 2. RANSAC
    # Use the FILTERED reference for the fitting
    mask = (~np.isnan(sig_dff)) & (~np.isnan(ref_filtered)) & (~np.isinf(sig_dff))

    if mask.sum() < 10:
        return sig_dff

    X = ref_filtered[mask].values.reshape(-1, 1)
    y = sig_dff[mask].values

    if np.all(X == X[0][0]):
        return sig_dff

    try:
        from sklearn import linear_model
    except ImportError:
        raise ImportError("scikit-learn is required for RANSAC method. Please install it or use method='wallace'.")

    ransac = linear_model.RANSACRegressor()
    ransac.fit(X, y)

    # Predict using the FILTERED reference
    ref_full = ref_filtered.fillna(0).values.reshape(-1, 1)
    pred_ref = ransac.predict(ref_full)

    # Subtract
    return sig_dff - pred_ref

def calculate_rolling_zscore(series: pd.Series, fps: float, win_sec: float = 20.0) -> pd.Series:
    """Datta-style final Z-score: 20s sliding window."""
    window = int(win_sec * fps)
    if window < 1: window = 1

    rolling_mu = series.rolling(window=window, center=True, min_periods=1).mean()
    rolling_std = series.rolling(window=window, center=True, min_periods=1).std()

    return (series - rolling_mu) / rolling_std

from scipy import stats


def calculate_wallace_dff(signal_trace: pd.Series, iso_trace: pd.Series) -> pd.Series:
    """
    Implements the linear fit subtraction method (Wallace et al., 2025).
    """
    # 1. Clean data (remove NaNs)
    mask = ~np.isnan(signal_trace) & ~np.isnan(iso_trace)
    if mask.sum() < 2:
        return signal_trace * 0 # Return zeros or NaNs if not enough data

    x = iso_trace[mask] # Iso (Reference)
    y = signal_trace[mask] # Signal (470/560)

    if np.all(x == x.iloc[0]):
        return signal_trace * 0

    # 2. Fit the Iso to the Signal
    # We ask: "How much of the Signal is just a scaled version of Iso?"
    slope, intercept, _, _, _ = stats.linregress(x, y)

    # 3. Create the "Predicted Artifact"
    # This is what the signal WOULD look like if it were only noise/bleaching
    predicted_artifact = slope * iso_trace + intercept

    # 4. Subtract and Normalize
    # (Raw - Artifact) / Artifact
    dff = (signal_trace - predicted_artifact) / predicted_artifact

    return dff * 100  # Return as true % change

def calculate_manual_dff(signal_trace: pd.Series, iso_trace: pd.Series) -> pd.Series:
    """
    Implements the 'Manual Method' (Simple Subtraction) from Wallace et al. (2025).
    
    Logic:
    1. Subtract the raw Isosbestic channel from the Signal channel.
    2. No regression fitting or scaling is performed.
    
    Note: The paper describes manual baseline adjustment in Clampfit.
    Here we implement the core mathematical operation: subtraction of the reference.
    """
    # 1. Clean data (remove NaNs)
    mask = ~np.isnan(signal_trace) & ~np.isnan(iso_trace)
    if mask.sum() < 2:
        return signal_trace * 0

    # 2. Simple Subtraction
    # We assume traces are already baseline-corrected or raw-aligned as per user preprocessing
    dff = signal_trace - iso_trace

    return dff

def calculate_signal_prominence(series: pd.Series, fps: float, win_sec: float = 5.0, quantile: float = 0.1) -> pd.Series:
    """
    Calculate 'Prominence' trace: Signal height above a local baseline.
    Baseline = Rolling quantile (default 10th percentile) over win_sec.
    """
    window = int(win_sec * fps)
    if window < 1: window = 1

    # Local baseline
    baseline = series.rolling(window=window, center=True, min_periods=1).quantile(quantile)

    # Prominence = Signal - Baseline
    return series - baseline

