# fiberphotometry/data/syncer.py

import warnings
import pandas as pd
from fiberphotometry import config

class SyncError(Exception):
    """Raised when synchronization cannot proceed due to invalid or missing data."""
    pass


def calculate_cpt_index(raw_df: pd.DataFrame) -> int:
    """Find the index label of the first ‘Set Blank Images’ event, or -1 if none."""
    if raw_df is None:
        raise SyncError("Raw DataFrame is None in calculate_cpt_index")
    if "Item_Name" not in raw_df.columns:
        raise SyncError("Column 'Item_Name' missing in raw DataFrame")
    return next(
        (idx for idx, item in raw_df["Item_Name"].items()
        if item == "Set Blank Images"),
        -1
    )


def sync_session(session) -> None:
    """Calculate sync time and align raw, TTL, and all photometry streams."""
    raw_df = session.dfs.get_data("raw")
    cpt_idx = calculate_cpt_index(raw_df)
    if cpt_idx < 0:
        return  # no sync event found
    
    # validate raw time column
    RAW_TIME = config.SYNC["raw_time_col"]
    if RAW_TIME not in raw_df.columns:
        raise SyncError(f"Raw time column '{RAW_TIME}' not found")

    session.cpt = cpt_idx
    session.sync_time = float(raw_df.at[cpt_idx, RAW_TIME])
    sync_all_streams(session)


def sync_all_streams(session) -> None:
    """Perform the single‐logic synchronization across TTL, raw, and photometry."""
    cfg      = config.SYNC
    raw_df   = session.dfs.get_data("raw")
    ttl_df   = session.dfs.get_data("ttl")
    ref_key  = f"phot_{cfg['reference_phot_freq']}"
    ref_df   = session.dfs.get_data(ref_key)

    sec_zero_col  = "sec_from_zero"
    sec_trial_col = "sec_from_trial_start"

    FREQS_USED = config.FREQS_USED

    # 1) Validate required DataFrames
    for name, df in (("raw", raw_df), ("ttl", ttl_df), (ref_key, ref_df)):
        if df is None:
            raise ValueError(f"{name!r} DataFrame is missing")
        if df.empty:
            raise ValueError(f"{name!r} DataFrame is empty")

    # 2) Column names
    raw_time   = cfg["raw_time_col"]
    ttl_time   = cfg["ttl_time_col"]
    phot_time  = cfg["phot_time_col"]

    # 3) Validate required columns
    for col, name, df in (
        (raw_time, "raw", raw_df),
        (ttl_time, "ttl", ttl_df),
        (phot_time, ref_key, ref_df)
    ):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in {name} DataFrame")

    # 4) TTL: seconds from zero
    try:
        start_ttl = float(ttl_df[ttl_time].iloc[0])
    except Exception as e:
        raise ValueError(f"Invalid TTL start timestamp: {e}")
    ttl_series = pd.to_numeric(ttl_df[ttl_time], errors="coerce")
    if ttl_series.isna().all():
        raise ValueError("All TTL timestamps failed to convert")
    ttl_df[sec_zero_col] = ttl_series - start_ttl

    # 5) Offset via reference photometry
    try:
        start_phot = float(ref_df[phot_time].iloc[0])
    except Exception as e:
        raise ValueError(f"Invalid photometry start timestamp for {ref_key}: {e}")
    offset = (start_ttl - start_phot) - session.sync_time

    # 6) Stamp raw
    raw_series = pd.to_numeric(raw_df[raw_time], errors="coerce")
    if raw_series.isna().all():
        raise ValueError("All raw timestamps failed to convert")
    raw_df[sec_zero_col]        = raw_series + offset
    raw_df[sec_trial_col] = raw_df[raw_time] - session.sync_time

    
    # 7) Stamp each photometry stream (optional)
    #for freq in cfg["frequencies"]:
    for freq in FREQS_USED:
        key = f"phot_{freq}"
        df  = session.dfs.get_data(key)
        if df is None or df.empty:
            warnings.warn(f"Skipping {key!r}: no data", UserWarning)
            continue
        if phot_time not in df.columns:
            warnings.warn(f"Skipping {key!r}: missing time column '{phot_time}'", UserWarning)
            continue

        try:
            first = float(df[phot_time].iloc[0])
        except Exception as e:
            raise ValueError(f"Invalid start timestamp for {key}: {e}")
        zero_series = pd.to_numeric(df[phot_time], errors="coerce")
        if zero_series.isna().all():
            raise ValueError(f"All timestamps failed to convert for {key}")
        df["sec_from_zero"]        = zero_series - first
        df["sec_from_trial_start"] = df["sec_from_zero"] - offset


    # 8) Always truncate, warning how many rows were cut
    lengths = {}
    for freq in FREQS_USED:
        key = f"phot_{freq}"
        df  = session.dfs.get_data(key)
        # if there's no DataFrame, treat its length as zero
        lengths[key] = len(df) if df is not None else 0
        
    if not lengths:
        warnings.warn("No photometry streams to truncate", UserWarning)
    else:
        n_min = min(lengths.values())
        for key, length in lengths.items():
            if length > n_min:
                removed = length - n_min
                warnings.warn(f"Truncating {key!r}: removed {removed} rows", UserWarning)
            session.dfs.data[key] = session.dfs.get_data(key).iloc[:n_min]