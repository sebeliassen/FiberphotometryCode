import gc
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from fpcode.common.paths import get_repo_root

# ==============================================================================
# CONFIGURATION
# ==============================================================================
REPO_ROOT = Path(str((get_repo_root() / 'compat/syllable_generation')))
DATA_DIR = (get_repo_root() / 'studies/replications_datta/datta/processing/dlight_scalar_corr')

# ==============================================================================
# HELPERS
# ==============================================================================
def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
    sys.stdout.flush()

def get_ram_gb():
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3
    except: return 0.0

def optimize_dataframe_inplace(df):
    """Shrinks RAM usage by converting strings to categories."""
    if "r" in df.columns:
        df["r"] = pd.to_numeric(df["r"], errors="coerce").astype("float32")

    cat_cols = ["genotype", "timescale", "brain_reg", "neural_agg", "feature",
                "shuffle_mode", "hemisphere", "fiber_num", "orig_letter", "animal_id", "signal"]

    for c in cat_cols:
        if c in df.columns and df[c].dtype == "object":
            df[c] = df[c].astype("category")

def process_and_save_stats(df_subset, pipeline_name, valid_meta):
    """Calculates 95% Quantiles for shuffles."""
    modes = df_subset["shuffle_mode"].unique().tolist()

    for mode in modes:
        log(f"   --- Generating Stats for: {pipeline_name} ({mode}) ---")
        sub = df_subset[df_subset["shuffle_mode"] == mode]

        base_cols = ["genotype", "timescale", "brain_reg", "neural_agg", "feature"]
        grp_cols = base_cols + valid_meta

        # 1. Population Mean (avoids pseudoreplication)
        shuffle_means = sub.groupby(["shuffle_index"] + grp_cols, observed=True)["r"].mean()

        # 2. Chance Bands
        chance_df = shuffle_means.abs().groupby(grp_cols, observed=True).quantile(0.95)
        p_dist_df = shuffle_means.unstack("feature")

        out_name = DATA_DIR / f"STATS_{pipeline_name}_{mode}.pkl"
        log(f"      💾 Saving {out_name.name}")
        pd.to_pickle((chance_df, p_dist_df), out_name)

        del sub, chance_df, p_dist_df, shuffle_means
        gc.collect()

# ==============================================================================
# PART A: DIGEST SHUFFLES (STATS)
# ==============================================================================
def digest_shuffles():
    log(f"{'='*40}")
    log("DIGESTING SHUFFLES (STATS)")
    log(f"{'='*40}")

    candidates = list(DATA_DIR.glob("datta_corr_SHUFFLE_raw_combined_*.parquet"))
    if not candidates:
        log("❌ No shuffle parquet found.")
        return
    shuff_path = max(candidates, key=lambda p: p.stat().st_mtime)

    t0 = time.time()
    log(f"🚀 Loading {shuff_path.name}...")
    df = pd.read_parquet(
        shuff_path, engine="pyarrow",
        filters=[("genotype", "!=", "UNK"), ("brain_reg", "!=", "UNK")]
    )
    log(f"✅ Loaded {len(df):,} rows in {time.time() - t0:.1f}s.")

    optimize_dataframe_inplace(df)

    desired_meta = ["hemisphere", "fiber_num", "orig_letter"]
    valid_meta = [c for c in desired_meta if c in df.columns]

    log("✂️ Splitting Pipelines...")
    is_datta = df["signal"].astype(str).str.contains("_datta")

    # Datta
    df_datta = df[is_datta].copy()
    if not df_datta.empty: process_and_save_stats(df_datta, "datta_ransac", valid_meta)
    del df_datta; gc.collect()

    # Standard
    df_std = df[~is_datta].copy()
    if not df_std.empty: process_and_save_stats(df_std, "standard_z", valid_meta)
    del df_std; gc.collect()

    del df; gc.collect()

# ==============================================================================
# PART B: DIGEST OBSERVED (THE MISSING LINK)
# ==============================================================================
def digest_observed():
    log(f"\n{'='*40}")
    log("DIGESTING OBSERVED DATA")
    log(f"{'='*40}")

    candidates = list(DATA_DIR.glob("datta_corr_OBSERVED_raw_*.parquet"))
    if not candidates:
        log("❌ No Observed parquet found.")
        return
    obs_path = max(candidates, key=lambda p: p.stat().st_mtime)

    t0 = time.time()
    log(f"🚀 Loading {obs_path.name}...")
    df = pd.read_parquet(
        obs_path, engine="pyarrow",
        filters=[("genotype", "!=", "UNK"), ("brain_reg", "!=", "UNK")]
    )
    log(f"✅ Loaded {len(df):,} rows.")
    optimize_dataframe_inplace(df)

    log("✂️ Splitting and Saving Observed Pipelines...")
    is_datta = df["signal"].astype(str).str.contains("_datta")

    # 1. Save Datta Observed
    df_datta = df[is_datta].copy()
    if not df_datta.empty:
        out = DATA_DIR / "datta_corr_OBSERVED_datta_ransac.pkl"
        df_datta.to_pickle(out)
        log(f"   💾 Saved {out.name}")

    # 2. Save Standard Observed
    df_std = df[~is_datta].copy()
    if not df_std.empty:
        out = DATA_DIR / "datta_corr_OBSERVED_standard_z.pkl"
        df_std.to_pickle(out)
        log(f"   💾 Saved {out.name}")

    del df, df_datta, df_std
    gc.collect()

if __name__ == "__main__":
    digest_shuffles()
    digest_observed()  # <--- Now runs this too!
    log("\n🏁 Done.")
