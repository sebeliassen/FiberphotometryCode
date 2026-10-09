"""
Digest Results: Processes raw Parquet outputs into statistical summaries.
- Calculates 95% Quantile thresholds from Shuffle distributions (Chance Bands)
- Prepares Observed correlations for plotting
- Saves summaries as Pickle files for rapid visualization
"""
import pandas as pd
import numpy as np
import time
from lib import config

def optimize_dataframe(df):
    """Reduces memory usage by categorical conversion."""
    if "r" in df.columns:
        df["r"] = pd.to_numeric(df["r"], errors="coerce").astype("float32")
    
    cat_cols = ["genotype", "timescale", "brain_reg", "neural_agg", "feature", 
                "shuffle_mode", "hemisphere", "fiber_num", "animal_id", "signal"]
    
    for c in cat_cols:
        if c in df.columns and df[c].dtype == "object":
            df[c] = df[c].astype("category")
    return df

def digest_shuffles():
    print("Digesting Shuffles...")
    
    # 1. FIND NEWEST SHUFFLE FILE
    candidates = list(config.OUTPUT_DIR.glob("*datta_SHUFFLE_raw_*.parquet"))
    if not candidates:
        print("No shuffle files found.")
        return
        
    shuf_path = max(candidates, key=lambda p: p.stat().st_mtime)
    print(f"Loading {shuf_path.name}...")
    
    df = pd.read_parquet(shuf_path)
    df = optimize_dataframe(df)
    
    # 2. CALC STATS
    # Group by everything except shuffle_index
    # We want 95% Quantile of ABSOLUTE correlation as the "Chance" threshold
    
    modes = df["shuffle_mode"].unique()
    
    for mode in modes:
        print(f"Processing Mode: {mode}")
        sub = df[df["shuffle_mode"] == mode]
        
        # Group cols: essentially all metadata keys
        grp_cols = ["genotype", "timescale", "brain_reg", "neural_agg", "feature", "hemisphere", "fiber_num", "drug", "orig_letter"]
        # Ensure cols exist
        grp_cols = [c for c in grp_cols if c in sub.columns]
        
        # Standard Datta Logic:
        # A. Mean across animals/sessions first? 
        # In `digest_shuffles.py`, they group by ["(metadata)"] and calculate quantile.
        # But wait, the raw shuffle file has (session_id, shuffle_index).
        # We need to aggregate across SESSIONS first to get the population mean for that shuffle iter.
        
        # 1. Population Mean per Shuffle Index
        # (This matches 'digest_shuffles.py' logic: "shuffle_means = sub.groupby(...)")
        
        pop_means = sub.groupby(["shuffle_index"] + grp_cols, observed=True)["r"].mean()
        
        # 2. 95% Quantile of Magnitude
        chance_df = pop_means.abs().groupby(grp_cols, observed=True).quantile(0.95)
        
        # 3. P-Value Distribution (Observed vs Shuffle Dist)
        # We save the means for plotting the gray bands
        # But actually, the gray band is usually just the max/min envelope or the quantile.
        # Let's save the indexed means for the 'envelope' plotting.
        
        p_dist_df = pop_means.unstack("feature") # [Index -> Feature Columns]
        
        # SAVE
        prefix = "DEBUG_" if "DEBUG" in shuf_path.name else ""
        out_name = config.OUTPUT_DIR / f"{prefix}STATS_{mode}.pkl"
        
        print(f"Saving {out_name.name}")
        pd.to_pickle((chance_df, p_dist_df), out_name)

def digest_observed():
    print("Digesting Observed...")
    candidates = list(config.OUTPUT_DIR.glob("*datta_OBSERVED_raw_*.parquet"))
    if not candidates:
        print("No observed files found.")
        return
        
    obs_path = max(candidates, key=lambda p: p.stat().st_mtime)
    print(f"Loading {obs_path.name}...")
    
    df = pd.read_parquet(obs_path)
    df = optimize_dataframe(df)
    
    # Save as Pickle for consistent loading
    prefix = "DEBUG_" if "DEBUG" in obs_path.name else ""
    out_name = config.OUTPUT_DIR / f"{prefix}OBSERVED.pkl"
    df.to_pickle(out_name)
    print(f"Saved {out_name.name}")

def main():
    digest_shuffles()
    digest_observed()

if __name__ == "__main__":
    main()
