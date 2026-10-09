"""
Shuffling Engine: Implementation of Permutation and Roll (circular shift) shuffles.
Includes worker functions for parallel execution.
"""
import numpy as np
import pandas as pd
from . import config, stats

def _shuffle_signals_within_groups(df, group_cols, cols_to_shuffle, seed, mode):
    """Core shuffling logic: either Permute (shuffle within group) or Roll (circular shift)."""
    out = df.copy()
    rs = np.random.RandomState(seed)
    
    if mode == "permute":
        for c in cols_to_shuffle:
            out[c] = (
                out.groupby(group_cols, observed=True)[c]
                .transform(lambda x: pd.Series(rs.permutation(x.values), index=x.index))
            )
        return out
        
    elif mode == "roll":
        grouped = out.groupby(group_cols, observed=True, sort=False)
        group_ids = grouped.ngroup().to_numpy()
        n_groups = group_ids.max() + 1
        arrays = {c: out[c].to_numpy() for c in cols_to_shuffle}
        
        for g in range(n_groups):
            idx = np.nonzero(group_ids == g)[0]
            if idx.size <= 1: continue
            
            # Random shift
            shift = rs.randint(0, idx.size)
            if shift == 0: continue
            
            for c in cols_to_shuffle:
                arr = arrays[c]
                arr[idx] = np.roll(arr[idx], shift)
                
        for c, arr in arrays.items():
            out[c] = arr
            
        return out
        
    return df

def run_shuffle_job(seed: int, mode: str, agg_df: pd.DataFrame, signal_cols: list[str]) -> pd.DataFrame:
    """Worker function for Parallel Execution."""
    
    shuffle_group = ["timescale", "session_id", "neural_agg", "scalar_agg"]
    
    # 1. Apply Shuffle
    df_shuffled = _shuffle_signals_within_groups(
        agg_df, 
        group_cols=shuffle_group, 
        cols_to_shuffle=signal_cols, 
        seed=seed, 
        mode=mode
    )
    
    # 2. Calculate Correlations on Shuffled Data
    # Reuse logic from stats.py!
    # Reuse `calculate_correlations` but we need to adapt it slightly 
    # because `calculate_correlations` expects to loop over `corr_group`.
    # It should work as is.
    
    blocks = stats.calculate_correlations(df_shuffled, signal_cols)
    
    if not blocks.empty:
        blocks["shuffle_index"] = seed
        blocks["shuffle_mode"] = mode
        return blocks
        
    return pd.DataFrame()
