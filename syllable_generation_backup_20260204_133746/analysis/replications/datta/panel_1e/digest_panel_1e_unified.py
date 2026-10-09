
"""
Unified Digest Results: Processes raw Parquet outputs into statistical summaries.
Supports both IFDN and AMPH projects.
"""
import sys
import argparse
import time
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

# Pathing
this_file = Path(__file__).resolve()
repo_root = this_file.parents[5] 
datta_dir = this_file.parents[1]

if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
if str(datta_dir) not in sys.path:
    sys.path.insert(0, str(datta_dir))

from lib import config

def optimize_dataframe(df):
    """Reduces memory usage by categorical conversion."""
    if "r" in df.columns:
        df["r"] = pd.to_numeric(df["r"], errors="coerce").astype("float32")
    
    cat_cols = ["genotype", "timescale", "brain_reg", "neural_agg", "feature", 
                "shuffle_mode", "hemisphere", "fiber_num", "animal_id", "signal", "drug", "orig_letter"]
    
    for c in cat_cols:
        if c in df.columns and df[c].dtype == "object":
            df[c] = df[c].astype("category")
    return df

def digest_shuffles(output_dir, shuffle_limit=None):
    print(f"Digesting Shuffles in {output_dir}...")
    
    candidates = list(output_dir.glob("*datta_SHUFFLE_raw_*.parquet"))
    if not candidates:
        print("No shuffle files found.")
        return
        
    shuf_path = max(candidates, key=lambda p: p.stat().st_mtime)
    print(f"Loading {shuf_path.name}...")
    
    df = pd.read_parquet(shuf_path)
    if shuffle_limit and "shuffle_index" in df.columns:
        print(f"Limiting to first {shuffle_limit} shuffles...")
        df = df[df["shuffle_index"] < shuffle_limit]
        
    df = optimize_dataframe(df)
    
    modes = df["shuffle_mode"].unique()
    
    for mode in modes:
        print(f"Processing Mode: {mode}")
        sub = df[df["shuffle_mode"] == mode]
        
        grp_cols = ["genotype", "timescale", "brain_reg", "neural_agg", "feature", "hemisphere", "fiber_num", "drug", "orig_letter"]
        grp_cols = [c for c in grp_cols if c in sub.columns]
        
        # 1. Population Mean per Shuffle Index
        pop_means = sub.groupby(["shuffle_index"] + grp_cols, observed=True)["r"].mean()
        
        # 2. 95% Quantile of Magnitude
        chance_df = pop_means.abs().groupby(grp_cols, observed=True).quantile(0.95)
        
        # 3. P-Value Distribution
        p_dist_df = pop_means.unstack("feature") 
        
        # SAVE
        prefix = "DEBUG_" if "DEBUG" in shuf_path.name else ""
        out_name = output_dir / f"{prefix}STATS_{mode}.pkl"
        
        print(f"Saving {out_name.name}")
        pd.to_pickle((chance_df, p_dist_df), out_name)

def digest_observed(output_dir):
    print(f"Digesting Observed in {output_dir}...")
    candidates = list(output_dir.glob("*datta_OBSERVED_raw_*.parquet"))
    if not candidates:
        print("No observed files found.")
        return
        
    obs_path = max(candidates, key=lambda p: p.stat().st_mtime)
    print(f"Loading {obs_path.name}...")
    
    df = pd.read_parquet(obs_path)
    df = optimize_dataframe(df)
    
    # Save as Pickle
    prefix = "DEBUG_" if "DEBUG" in obs_path.name else ""
    out_name = output_dir / f"{prefix}OBSERVED.pkl"
    df.to_pickle(out_name)
    print(f"Saved {out_name.name}")

def main():
    parser = argparse.ArgumentParser(description="Unified Results Digestion")
    parser.add_argument("--project", type=str, required=True, choices=["ifdn", "amph"])
    parser.add_argument("--output-dir", type=str, default=None)
    
    parser.add_argument("--shuffle-limit", type=int, default=None)
    
    args = parser.parse_args()
    
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        # Resolve from config
        if args.project == "ifdn":
            out_dir = config.PANEL_DIRS.get("1e", config.OUTPUT_DIR)
        else:
            out_dir = config.PANEL_DIRS.get("1e_amph", config.OUTPUT_DIR)
            
    digest_shuffles(out_dir, shuffle_limit=args.shuffle_limit)
    digest_observed(out_dir)

if __name__ == "__main__":
    main()
