"""
Statistical Utilities: Z-scoring, temporal binning, and Pearson correlation calculations.
"""
import numpy as np
import pandas as pd

from . import config


def _zscore_custom(x):
    """Robust z-score helper."""
    v = x.values.astype(float)
    m = np.nanmean(v)
    s = np.nanstd(v)
    if not np.isfinite(s) or s == 0:
        return (v - m)
    return (v - m) / s

def zscore_groupwise(df: pd.DataFrame, by: str, cols: list[str]) -> pd.DataFrame:
    """Applies Z-Score to selected columns within groups."""
    out = df.copy()
    out[cols] = out.groupby(by, observed=True)[cols].transform(_zscore_custom)
    return out

def _time_bin_index(ts: pd.Series, sid: pd.Series, width_s: float) -> pd.Series:
    """Calculates bin index for each timestamp based on session start."""
    # We need t0 for each session
    # Optimization: Use transform('min')
    t0 = ts.groupby(sid, observed=True).transform("min")
    return np.floor((ts - t0) / float(width_s)).astype("int32")

def bin_data(data: pd.DataFrame, timescale: float | str, signal_cols: list[str], neural_agg_func, scalar_agg_func="mean") -> pd.DataFrame:
    """Aggregates data into time bins or syllable blocks."""
    scalar_cols = config.SCALAR_COLS

    # Determine Bin Index
    if timescale == "syllable":
        time_bin = data["syllable_number"].astype("int32")
    else:
        time_bin = _time_bin_index(data["timestamp"], data["session_id"], timescale)

    grp_keys = [data["session_id"], data["animal_id"], time_bin]

    # 1. Scalar Aggregation
    data_keys = scalar_cols + ["labels"]
    x = data.groupby(grp_keys, observed=True)[data_keys].agg(scalar_agg_func)

    # 2. Neural Aggregation
    y = data.groupby(grp_keys, observed=True)[signal_cols].agg(neural_agg_func)

    agg_matrix = x.join(y)

    # 3. Filter by Modal Duration (for time bins only)
    if timescale != "syllable":
        sz = data.groupby(grp_keys, observed=True).size()
        if not sz.empty:
            modal_sz = sz.mode().iat[0]
            # Keep only bins that have the "standard" number of frames (avoids edge effects)
            valid_bins = sz[sz == modal_sz].index
            agg_matrix = agg_matrix.loc[valid_bins]

    # Add Metadata
    agg_matrix["timescale"]   = -5 if timescale == "syllable" else int(timescale)
    agg_matrix["neural_agg"]  = getattr(neural_agg_func, "__name__", str(neural_agg_func))
    agg_matrix["scalar_agg"]  = getattr(scalar_agg_func, "__name__", str(scalar_agg_func))

    return agg_matrix
def run_bin_block(data, signal_cols, neural_agg_func, scalar_agg_func, timescales):
    """Orchestrates binning for a set of timescales and a neural agg function."""
    mats = []
    for ts in timescales:
        mats.append(bin_data(data, ts, signal_cols, neural_agg_func, scalar_agg_func))
    # Always include syllable scale
    mats.append(bin_data(data, "syllable", signal_cols, neural_agg_func, scalar_agg_func))
    return pd.concat(mats, axis=0)

def calculate_correlations(agg_df: pd.DataFrame, signal_cols: list[str]) -> pd.DataFrame:
    """Calculates Pearson correlation between Scalars and Signals within groups."""
    corr_group = ["timescale", "session_id", "neural_agg", "scalar_agg"]
    corr_keys  = config.SCALAR_COLS + ["labels"] + signal_cols

    use_agg = agg_df.dropna(subset=corr_keys).sort_values(["timescale", "session_id", "animal_id", "bin"]).reset_index(drop=True)

    blocks = []

    # Main Loop
    for grp_vals, sub in use_agg.groupby(corr_group, observed=True):
        # Pearson Corr Matrix
        C = sub[corr_keys].corr(method=config.CORR_METHOD)

        tall_rows = []
        for sig in signal_cols:
            if sig not in C.index: continue

            # Check all scalar features
            for sk in config.SCALAR_COLS + ["labels"]:
                if sk in C.columns:
                    r_val = C.loc[sk, sig]
                    if pd.notna(r_val):
                        tall_rows.append({"signal": sig, "feature": sk, "r": r_val})

        blk = pd.DataFrame(tall_rows)
        if not blk.empty:
            for k, v in zip(corr_group, grp_vals): blk[k] = v
            blocks.append(blk)

    if blocks:
        return pd.concat(blocks, ignore_index=True)
    return pd.DataFrame()
