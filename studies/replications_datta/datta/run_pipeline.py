"""
Datta Pipeline: End-to-end processing of fiber photometry correlations.
1. Loads session data
2. Calculates kinematics (acceleration, jerk, transition, duration)
3. Aggregates data into temporal bins (1s..60s) and syllable blocks
4. Computes Pearson correlations (observed)
5. Performs statistical shuffling (Roll and Permute) for significance testing
6. Saves results to Parquet files
"""
import pickle
import sys
import time

import pandas as pd
from joblib import Parallel, delayed

# Local Lib
from lib import config, features, shuffling, stats

# FIX: Add parent of REPO_ROOT to sys.path so pickle can find 'syllable_generation'
if str(config.REPO_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(config.REPO_ROOT.parent))

def load_sessions():
    """Loads sessions from the newest pickle in the data directory."""
    print("Searching for session pickles...")
    data_dir = config.REPO_ROOT / "data"
    candidates = list(data_dir.glob("*sessions*all_data*.pkl")) + list(data_dir.glob("*.pkl"))

    if not candidates:
        raise FileNotFoundError(f"Could not find sessions pickle in {data_dir}")

    pkl_path = max(candidates, key=lambda p: p.stat().st_mtime)
    print(f"Loading sessions from: {pkl_path}")

    with pkl_path.open("rb") as f:
        sessions = pickle.load(f)

    # Filter valid sessions
    valid = [
        s for s in sessions
        if hasattr(s, "all_data_df") and isinstance(s.all_data_df, pd.DataFrame) and not s.all_data_df.empty
    ]
    print(f"Loaded {len(valid)} usable sessions.")
    return valid

def prepare_data(sessions):
    """Converts session objects to a long-form DataFrame with features."""
    rows = []
    sig_annot_rows = []

    print("Extracting features from sessions...")

    # DEBUG: Limit sessions
    if config.DEBUG_MODE:
        print("[DEBUG] Subsetting to first 2 sessions.")
        sessions = sessions[:2]

    for s in sessions:
        df = s.all_data_df.copy()
        meta = s.metadata or {}

        # 1. Feature Extraction (Kinematics)
        df = features.calculate_kinematics(df)
        if "acceleration" not in df.columns: continue # Skip if failed

        # 2. Metadata Injection
        df["session_id"] = s.session_id
        df["animal_id"] = meta.get("animal_id", "UNK")
        df["genotype"] = meta.get("genotype", "UNK")

        # 3. Signal Extraction Metadata
        sig_meta = features.extract_signal_metadata(df, meta)
        if not sig_meta: continue

        for sm in sig_meta:
            sm["session_id"] = s.session_id
            sm["animal_id"] = meta.get("animal_id", "UNK")
            sm["genotype"] = meta.get("genotype", "UNK")
            sig_annot_rows.append(sm)

        keep_signals = [x["signal"] for x in sig_meta]

        # 4. Select Columns
        keep_cols = ([
            "session_id", "animal_id", "genotype", "timestamp",
            "labels", "transition", "syllable_number",
            "velocity_px_s", "angular_velocity",
            "acceleration", "jerk", "duration"
        ] + keep_signals)

        # Ensure numeric signals
        for c in keep_signals:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        rows.append(df[keep_cols])

    if not rows:
        raise RuntimeError("No valid data found.")

    use_data = pd.concat(rows, ignore_index=True)

    # 5. Cleanup
    use_data = use_data.dropna(subset=["timestamp"])

    # 6. Z-Score (Groupwise)
    # We Z-Score signals AND scalars within session
    zscore_cols = [c for c in use_data.columns if c.startswith("signal")] + config.SCALAR_COLS
    use_data = stats.zscore_groupwise(use_data, by="session_id", cols=zscore_cols)

    return use_data, pd.DataFrame(sig_annot_rows).drop_duplicates()

def clean_for_parquet(df):
    """Converts object columns to string to avoid Parquet errors."""
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str)
    return df

def main():
    start_time = time.time()

    # 1. LOAD & PREPARE
    sessions = load_sessions()
    use_data, annot_df = prepare_data(sessions)
    print(f"Data Prepared. Shape: {use_data.shape}")

    signal_cols = [c for c in use_data.columns if c.startswith("signal")]

    # 2. AGGREGATION
    print("Running Aggregation (Binning)...")

    delays = []
    # Block 1: Peak Rate Cross
    delays.append(delayed(stats.run_bin_block)(use_data, signal_cols, features.peak_rate_cross, "mean", config.BIN_GRID))
    # Block 2: Mean
    delays.append(delayed(stats.run_bin_block)(use_data, signal_cols, "mean", "mean", config.BIN_GRID))

    agg_mats = Parallel(n_jobs=config.NJOBS, verbose=5)(delays)

    agg_df = pd.concat(agg_mats).reset_index().rename(columns={"level_0": "session_id", "level_1": "animal_id", "level_2": "bin"})
    agg_df["timescale"] = agg_df["timescale"].astype(int)

    print(f"Aggregation Complete. Shape: {agg_df.shape}")

    # 2.5 Ensure Deterministic Order for Shuffles (Crucial for parity)
    # Shuffling with sort=False in groupby depends on row order for random shifts
    agg_df = agg_df.sort_values(["timescale", "session_id", "animal_id", "bin"]).reset_index(drop=True)

    # 3. OBSERVED CORRELATIONS
    print("Calculating Observed Correlations...")
    # NOTE: calculate_correlations handles dropna itself, but for shuffles we want to
    # ensure we pass the same filtered set as legacy if we want bit-parity.
    # Legacy drops NaNs from the total set before shuffling.
    corr_keys = config.SCALAR_COLS + ["labels"] + signal_cols
    agg_df = agg_df.dropna(subset=corr_keys).reset_index(drop=True)

    # 3.5 Sort Signal Cols (Crucial for Permute Parity)
    signal_cols = sorted(signal_cols)

    corr_df = stats.calculate_correlations(agg_df, signal_cols)
    print(f"Observed Correlations: {corr_df.shape}")

    # 4. SHUFFLING
    print(f"Running Shuffles (N={config.NSHUFF})...")
    shuffle_tasks = []

    for i in range(config.NSHUFF):
        shuffle_tasks.append(delayed(shuffling.run_shuffle_job)(i, "roll", agg_df, signal_cols))
        shuffle_tasks.append(delayed(shuffling.run_shuffle_job)(i, "permute", agg_df, signal_cols))

    shuffle_results = Parallel(n_jobs=config.NJOBS, verbose=5)(shuffle_tasks)
    shuffle_df = pd.concat([d for d in shuffle_results if not d.empty], ignore_index=True)
    print(f"Shuffles Complete. Shape: {shuffle_df.shape}")

    # 5. MERGE METADATA
    corr_df = corr_df.merge(annot_df, how="left", on=["session_id", "signal"])
    shuffle_df = shuffle_df.merge(annot_df, how="left", on=["session_id", "signal"])

    # 6. SAVE
    ts_str = str(int(time.time()))
    prefix = "DEBUG_" if config.DEBUG_MODE else ""

    agg_out = config.OUTPUT_DIR / f"{prefix}datta_AGG_raw_{ts_str}.parquet"
    obs_out = config.OUTPUT_DIR / f"{prefix}datta_OBSERVED_raw_{ts_str}.parquet"
    shuf_out = config.OUTPUT_DIR / f"{prefix}datta_SHUFFLE_raw_{ts_str}.parquet"

    print("Saving to Parquet...")
    clean_for_parquet(agg_df).to_parquet(agg_out, index=False)
    clean_for_parquet(corr_df).to_parquet(obs_out, index=False)
    clean_for_parquet(shuffle_df).to_parquet(shuf_out, index=False)

    print(f"Done in {time.time() - start_time:.1f}s")

if __name__ == "__main__":
    main()
