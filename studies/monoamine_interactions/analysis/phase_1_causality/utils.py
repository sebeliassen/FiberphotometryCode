
from typing import Optional, Tuple

import numpy as np
import pandas as pd


def check_stationarity(timeseries: np.ndarray, threshold: float = 0.05) -> Tuple[bool, float]:
    """
    Checks stationarity using the Augmented Dickey-Fuller test.
    Returns (is_stationary, p_value).
    """
    from statsmodels.tsa.stattools import adfuller

    # Handle NaNs or Infs
    valid = timeseries[np.isfinite(timeseries)]
    if len(valid) < 20:
        return False, 1.0

    try:
        result = adfuller(valid)
        p_value = result[1]
        return p_value < threshold, p_value
    except Exception as e:
        print(f"[WARN] Stationarity check failed: {e}")
        return False, 1.0

def get_paired_traces(
    session,
    region_source: str,
    region_target: str,
    signal_key: str = "470", # e.g. '470' or 'dff'
    zscore: bool = True,
    trim_start: int = 100, # Remove artifact-heavy start
    max_len: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts aligned trace pairs (source, target) from a session.
    
    Args:
        session: The Session object.
        region_source: Name of source region (must match metadata label or column).
        region_target: Name of target region.
        signal_key: The photometry dictionary key ('470', 'dff', etc.)
        zscore: If True, looks for '{col}_zscore', else use raw/dff column.
        trim_start: Frames to drop from start.
        max_len: Truncate to this length if set.

    Returns:
        (source_trace, target_trace, time_axis)
    """
    if not hasattr(session, "photometry") or signal_key not in session.photometry:
        raise ValueError(f"Session {session.session_id} missing {signal_key} data")

    df = session.photometry[signal_key]
    meta = session.metadata

    # Resolve region names to dataframe columns
    # We built a map in build_sessions, but let's be robust
    # Map: "VTA_NE" -> "signal_0" (if that's the labeling)

    # Helper to find column for a region name
    def resolve_col(region_name):
        # 1. Direct match
        if region_name in df.columns:
            return region_name

        # 2. Check metadata keys (e.g. signal_0="VTA_NE")
        # We look for keys starting with 'signal_'
        for k, v in meta.items():
            if isinstance(k, str) and k.startswith("signal_") and v == region_name:
                # This key 'signal_0' corresponds to the region.
                # The dataframe column might be 'signal_0' or 'signal_0_zscore'
                return k
        return None

    col_src_base = resolve_col(region_source)
    col_tgt_base = resolve_col(region_target)

    if not col_src_base or not col_tgt_base:
        all_regions = [v for k,v in meta.items() if str(k).startswith("signal_")]
        raise ValueError(f"Could not resolve regions '{region_source}', '{region_target}' in session {session.session_id}. Available: {all_regions}")

    # Suffix handling
    suffix = "_zscore" if zscore else ""
    # However, if columns were renamed in previous steps, they might not stick to standard schema across all project types.
    # But in *this* project (monoamine), we kept 'signal_0', 'signal_1' etc and just updated metadata.
    # Let's check if we need to append suffix

    col_src = f"{col_src_base}{suffix}"
    col_tgt = f"{col_tgt_base}{suffix}"

    if col_src not in df.columns:
        # Fallback: maybe the column IS the region name if renamed?
        # Or maybe zscore isn't there.
        # Check if base exists
        if col_src_base in df.columns:
            print(f"[WARN] {col_src} not found, using {col_src_base}")
            col_src = col_src_base
        else:
             raise ValueError(f"Column {col_src} (and base {col_src_base}) not found in DF")

    if col_tgt not in df.columns:
        if col_tgt_base in df.columns:
             print(f"[WARN] {col_tgt} not found, using {col_tgt_base}")
             col_tgt = col_tgt_base
        else:
             raise ValueError(f"Column {col_tgt} not found in DF")

    # Extract
    src_data = df[col_src].values[trim_start:]
    tgt_data = df[col_tgt].values[trim_start:]

    # Time axis (optional, create generic if missing)
    if 'time' in df.columns:
        t_data = df['time'].values[trim_start:]
    else:
        t_data = np.arange(len(src_data))

    # Truncate
    if max_len:
        src_data = src_data[:max_len]
        tgt_data = tgt_data[:max_len]
        t_data = t_data[:max_len]

    # Force float conversion
    try:
        src_data = src_data.astype(float)
        tgt_data = tgt_data.astype(float)
    except ValueError as e:
        print(f"[ERR] Could not convert to float: {e}")
        # Try convert objects with 'pd.to_numeric' equivalent if needed,
        # but let's assume it should be numeric.
        raise

    # Nan check
    valid_mask = np.isfinite(src_data) & np.isfinite(tgt_data)
    if not np.all(valid_mask):
        # Interpolate or fail? For causality, gaps are bad.
        # Let's just warn and linearly interpolate for now as a naive fix
        # Real fix: data should be clean coming in.
        # print(f"[WARN] NaNs detected in {session.session_id}, interpolating...")

        # Simple pandas interpolation
        tmp_df = pd.DataFrame({'s': src_data, 't': tgt_data})
        tmp_df = tmp_df.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
        src_data = tmp_df['s'].values
        tgt_data = tmp_df['t'].values

    return src_data, tgt_data, t_data
