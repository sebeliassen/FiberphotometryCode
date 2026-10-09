"""
Datta Pipeline (AMPH): End-to-end processing of fiber photometry correlations for AMPH dataset (Panel 1e).

This script is the primary driver for replicating the "Panel 1e" scalar analysis on the Amphetamine dataset.
It extends the baseline logic (`run_pipeline.py`) with specific handling for:
1.  **Drug Metadata**: Extracts 'drug' (Saline/Amphetamine) from session metadata and merges it into the analysis.
2.  **Parallel Aggregation**: Uses `joblib` to parallelize the heavy binning and shuffling steps across 32 cores.
3.  **Deterministic Sorting**: Ensures 100% reproducible shuffle results by strictly sorting dataframes before randomization.

Pipeline Stages:
1.  Load Sessions (`jonathan_amph_sessions_w_sylls.pkl`)
2.  Feature Engineering (Kinematics: Acceleration, Jerk) + Signal Metadata Extraction
3.  Z-Scoring (Groupwise per session)
4.  Parallel Binning (Timescales 1..60s)
5.  Observed Correlation Calculation
6.  Parallel Shuffling (Roll + Permute x 300 iterations)
7.  Parquet Output Generation (Aggregated, Observed, Shuffled)
"""
import pickle
import sys
import time

import pandas as pd
from joblib import Parallel, delayed
from lib import config, features, shuffling, stats

# Path fix for pickle
PARENT_CODE_DIR = config.REPO_ROOT.parent
if str(PARENT_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_CODE_DIR))

def load_sessions():
    """Loads AMPH sessions."""
    print("Searching for AMPH session pickles...")
    data_dir = config.REPO_ROOT / "data"
    candidates = list(data_dir.glob("*amph*w_sylls.pkl"))

    if not candidates:
        raise FileNotFoundError(f"Could not find AMPH sessions pickle in {data_dir}")

    pkl_path = max(candidates, key=lambda p: p.stat().st_mtime)
    print(f"Loading sessions from: {pkl_path}")

    with pkl_path.open("rb") as f:
        sessions = pickle.load(f)

    valid = [
        s for s in sessions
        if hasattr(s, "all_data_df") and isinstance(s.all_data_df, pd.DataFrame) and not s.all_data_df.empty
    ]
    print(f"Loaded {len(valid)} usable sessions.")
    return valid

def prepare_data(sessions):
    """Deep copy of run_pipeline.prepare_data with Drug extraction."""
    rows = []
    sig_annot_rows = []
    session_drug_map = {}

    print("Extracting features from sessions...")

    if config.DEBUG_MODE:
        print("[DEBUG] Subsetting...")
        sessions = sessions[:2]

    for s in sessions:
        df = s.all_data_df.copy()
        meta = s.metadata or {}

        # Drug Extraction
        drug = meta.get("drug", "UNK")
        session_drug_map[s.session_id] = drug

        df = features.calculate_kinematics(df)
        if "acceleration" not in df.columns: continue

        df["session_id"] = s.session_id
        df["animal_id"] = meta.get("animal_id", "UNK")
        df["genotype"] = meta.get("genotype", "UNK")
        df["drug"] = drug

        sig_meta = features.extract_signal_metadata(df, meta)
        if not sig_meta: continue

        for sm in sig_meta:
            sm["session_id"] = s.session_id
            sm["animal_id"] = meta.get("animal_id", "UNK")
            sm["genotype"] = meta.get("genotype", "UNK")
            sm["drug"] = drug
            sig_annot_rows.append(sm)

        keep_signals = [x["signal"] for x in sig_meta]

        # Keep Cols (Include labels/drug)
        keep_cols = ([
            "session_id", "animal_id", "genotype", "drug", "timestamp",
            "labels", "transition", "syllable_number",
            "velocity_px_s", "angular_velocity",
            "acceleration", "jerk", "duration"
        ] + keep_signals)

        for c in keep_signals:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        # Ensure Timestamps are valid
        df = df.dropna(subset=["timestamp"])

        rows.append(df[keep_cols])

    if not rows: raise RuntimeError("No valid data.")
    return pd.concat(rows, ignore_index=True), sig_annot_rows, session_drug_map

def main():
    start_time = time.time()

    # 1. Setup Output
    config.OUTPUT_DIR = config.PANEL_DIRS["1e_amph"]
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Load
    sessions = load_sessions()

    # 3. Features
    use_data, sig_annot_rows, drug_map = prepare_data(sessions)
    print(f"Data Prepared. Shape: {use_data.shape}")

    # 4. Z-Score
    signal_cols = sorted(list(set([m['signal'] for m in sig_annot_rows])))
    scalar_cols = config.SCALAR_COLS
    cols_to_z = [c for c in signal_cols + scalar_cols if c in use_data.columns]

    print("Z-Scoring...")
    use_data = stats.zscore_groupwise(use_data, "session_id", cols_to_z)

    # 5. Binning
    print("Running Aggregation (Binning)...")

    delays = []
    # Block 1: Peak Rate Cross
    delays.append(delayed(stats.run_bin_block)(use_data, signal_cols, features.peak_rate_cross, "mean", config.BIN_GRID))
    # Block 2: Mean
    delays.append(delayed(stats.run_bin_block)(use_data, signal_cols, "mean", "mean", config.BIN_GRID))

    agg_mats = Parallel(n_jobs=config.NJOBS, verbose=5)(delays)

    agg_df = pd.concat(agg_mats, axis=0)

    # Standard Datta Reset/Rename to ensure 'bin' exists
    # If the levels are named, reset_index preserves names. If not, it uses level_N.
    agg_df = agg_df.reset_index()
    rename_map = {"level_0": "session_id", "level_1": "animal_id", "level_2": "bin"}
    # Also handle if they were named but we want to standardize
    for old, new in [("session_id", "session_id"), ("animal_id", "animal_id"), ("syllable_number", "bin"), ("level_2", "bin")]:
        if old in agg_df.columns and old != new:
            rename_map[old] = new

    agg_df = agg_df.rename(columns=rename_map)

    # Re-merge Drug/Genotype
    meta_df = use_data[["session_id", "animal_id", "genotype", "drug"]].drop_duplicates()
    cols_to_merge = [c for c in ["genotype", "drug"] if c not in agg_df.columns]
    if cols_to_merge:
        agg_df = agg_df.merge(meta_df[["session_id", "animal_id"] + cols_to_merge], on=["session_id", "animal_id"], how="left")

    agg_df["timescale"] = agg_df["timescale"].astype(int)

    # 5.5 Ensure Deterministic Order for Shuffles
    agg_df = agg_df.sort_values(["timescale", "session_id", "animal_id", "bin"]).reset_index(drop=True)

    print(f"Aggregation Complete. Shape: {agg_df.shape}")

    # 6. Correlations
    print("Calculating Observed Correlations...")
    # Prep keys for dropna
    corr_keys = config.SCALAR_COLS + ["labels"] + signal_cols
    agg_df = agg_df.dropna(subset=corr_keys).reset_index(drop=True)

    corr_df = stats.calculate_correlations(agg_df, signal_cols)
    print(f"Observed Correlations: {corr_df.shape}")

    # Merge Metadata into Correlations
    corr_df = corr_df.merge(meta_df[["session_id", "animal_id", "genotype", "drug"]], on="session_id", how="left")

    # 7. Shuffling
    print(f"Running Shuffles (N={config.NSHUFF})...")
    shuffle_tasks = []
    for i in range(config.NSHUFF):
        shuffle_tasks.append(delayed(shuffling.run_shuffle_job)(i, "roll", agg_df, signal_cols))
        shuffle_tasks.append(delayed(shuffling.run_shuffle_job)(i, "permute", agg_df, signal_cols))

    shuffle_results = Parallel(n_jobs=config.NJOBS, verbose=5)(shuffle_tasks)
    shuff_df = pd.concat([d for d in shuffle_results if not d.empty], ignore_index=True)
    print(f"Shuffles Complete. Shape: {shuff_df.shape}")

    # Merge Meta into Shuffles
    shuff_df = shuff_df.merge(meta_df[["session_id", "animal_id", "genotype", "drug"]], on="session_id", how="left")

    # 8. Save
    ts_str = str(int(time.time()))
    prefix = "DEBUG_" if config.DEBUG_MODE else ""

    # Merge Signal Annotations (Region/Hemisphere) into long-form outputs?
    # Usually digest does this. But standard datta pipeline merges sig_annot later in plot.
    # We'll save sig_annot via the 'signal' column matching?
    # No, we assume standard naming.
    # But digest needs to know Region.
    # Let's save a "metadata_map" or just ensure obs_raw has it?
    # run_pipeline saves raw, digest reads raw, digest merges sig_annot?
    # digest_results.py: "def digest_observed... uses standard global map?"
    # No, digest_results RE-LOADS sessions? No.
    # It seems to infer from filename or requires signal metadata.
    # Actually plot_results.py has `apply_color_hotfix`.
    # Let's just save the raw dataframes. They contain session_id/signal/value.
    # We should merge the generic signal metadata NOW to be safe?
    # run_pipeline merges `sig_annot` at the end!

    sig_annot = pd.DataFrame(sig_annot_rows).drop_duplicates()
    # Merge on [session_id, signal]
    if "session_id" in corr_df.columns and "signal" in corr_df.columns:
        corr_df = corr_df.merge(sig_annot, on=["session_id", "signal"], how="left", suffixes=("", "_meta"))
    if "session_id" in shuff_df.columns and "signal" in shuff_df.columns:
        shuff_df = shuff_df.merge(sig_annot, on=["session_id", "signal"], how="left", suffixes=("", "_meta"))

    agg_out = config.OUTPUT_DIR / f"{prefix}datta_AGG_raw_{ts_str}.parquet"
    obs_out = config.OUTPUT_DIR / f"{prefix}datta_OBSERVED_raw_{ts_str}.parquet"
    shuf_out = config.OUTPUT_DIR / f"{prefix}datta_SHUFFLE_raw_{ts_str}.parquet"

    def clean_for_parquet(df):
        for col in df.select_dtypes(include=['object', 'category']).columns:
            df[col] = df[col].astype(str)
        return df

    print("Saving to Parquet...")
    clean_for_parquet(agg_df).to_parquet(agg_out, index=False)
    clean_for_parquet(corr_df).to_parquet(obs_out, index=False)
    clean_for_parquet(shuff_df).to_parquet(shuf_out, index=False)

    print(f"Done in {time.time() - start_time:.1f}s")

if __name__ == "__main__":
    main()
