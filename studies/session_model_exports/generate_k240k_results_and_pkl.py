#!/usr/bin/env python3
"""
generate_k240k_results_and_pkl.py

Extracts results.h5 from karen_project_2/final_karen_model_k240k/checkpoint.h5,
computes kinematics via compute_moseq_df, and weaves the syllables onto the 80Hz
timeline to produce the enriched session pickle.
"""

from __future__ import annotations

import gc
import logging
import pickle
import sys
import time
from pathlib import Path

# Ensure numpy 2 backward compatibility for bokeh / kpms
import numpy as np

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import keypoint_moseq as kpms

from fpcode.photometry.alignment import build_aligned_moseq_phot
from syllable_generation.analysis.utils.sampling import build_aligned_dfs, flatten_results_to_timeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("generate_k240k")

PROJECT_DIR = REPO_ROOT / "syllable_generation" / "syllabizers" / "keypoint_moseq_syll" / "karen_project_2"
MODEL_NAME = "final_karen_model_k240k"
SESSIONS_IN = REPO_ROOT / "syllable_generation" / "data" / "sessions_karen_project_dirty.pkl"
SESSIONS_OUT_K240K = REPO_ROOT / "syllable_generation" / "data" / "sessions_karen_project_w_sylls_k240k.pkl"
SESSIONS_OUT_MAIN = REPO_ROOT / "syllable_generation" / "data" / "sessions_karen_project_w_sylls.pkl"


def main():
    t0_total = time.time()
    model_dir = PROJECT_DIR / MODEL_NAME
    results_path = model_dir / "results.h5"
    vfi_path = model_dir / "video_frame_indexes.npz"

    log.info("========================================================================")
    log.info("=== Generating results.h5 and Syllable PKL for %s ===", MODEL_NAME)
    log.info("========================================================================")

    # 1. GENERATE RESULTS.H5
    if results_path.exists():
        log.info("Found existing results.h5 at %s. Loading...", results_path)
        results = kpms.load_results(str(PROJECT_DIR), MODEL_NAME)
    else:
        log.info("Step 1: results.h5 not found. Extracting from checkpoint.h5...")

        # Try reindexing syllables by frequency (most frequent = 0)
        try:
            log.info("Reindexing syllables in checkpoint by usage frequency...")
            kpms.reindex_syllables_in_checkpoint(str(PROJECT_DIR), MODEL_NAME)
            log.info("Reindexing complete.")
        except Exception as e:
            log.warning("Reindexing skipped/failed: %s", e)

        log.info("Loading checkpoint from %s...", model_dir)
        t_load = time.time()
        model, data, metadata, current_iter = kpms.load_checkpoint(str(PROJECT_DIR), MODEL_NAME)
        log.info("Loaded checkpoint at iteration %s in %.1fs.", current_iter, time.time() - t_load)

        log.info("Extracting results and saving to %s...", results_path)
        t_ext = time.time()
        results = kpms.extract_results(model, metadata, str(PROJECT_DIR), MODEL_NAME, save_results=True)
        log.info("results.h5 saved successfully in %.1fs (%d parts).", time.time() - t_ext, len(results))

        del model, data, metadata
        gc.collect()

    # 2. LOAD VIDEO FRAME INDEXES
    log.info("Step 2: Loading video_frame_indexes from %s...", vfi_path)
    with np.load(vfi_path) as z:
        vfi = {k: z[k].astype(int) for k in z.files}
    log.info("Loaded VFI for %d recordings.", len(vfi))

    # 3. COMPUTE MOSEQ DF (Kinematics)
    log.info("Step 3: Computing kinematics via compute_moseq_df...")
    index_csv = PROJECT_DIR / "index.csv"
    if not index_csv.exists():
        log.info("Creating index.csv...")
        df_index = pd.DataFrame({"name": list(results.keys()), "group": ["default"] * len(results)})
        df_index.to_csv(index_csv, index=False)

    t_moseq = time.time()
    moseq_df = kpms.compute_moseq_df(str(PROJECT_DIR), MODEL_NAME, smooth_heading=True)
    log.info("moseq_df computed in %.1fs (%d rows, %d cols).", time.time() - t_moseq, len(moseq_df), len(moseq_df.columns))

    if "onset" not in moseq_df.columns and "syllable" in moseq_df.columns:
        moseq_df["onset"] = moseq_df.groupby("name")["syllable"].transform(lambda x: x.diff() != 0)
        moseq_df.loc[moseq_df.groupby("name").head(1).index, "onset"] = True

    if index_csv.exists():
        groups_df = pd.read_csv(index_csv)
        if "name" in groups_df.columns and "group" in groups_df.columns:
            name_to_group = dict(zip(groups_df["name"], groups_df["group"]))
            moseq_df["group"] = moseq_df["name"].map(name_to_group)

    common_keys = sorted(list(set(results.keys()) & set(vfi.keys())))
    log.info("Common keys across results and VFI: %d recordings.", len(common_keys))
    results = {k: results[k] for k in common_keys}
    vfi = {k: vfi[k] for k in common_keys}
    moseq_df = moseq_df[moseq_df["name"].isin(common_keys)].copy()

    # 4. LOAD SESSIONS AND MAP
    log.info("Step 4: Loading base sessions from %s...", SESSIONS_IN)
    with open(SESSIONS_IN, "rb") as f:
        all_sessions = pickle.load(f)
    log.info("Loaded %d sessions.", len(all_sessions))

    r2s = {}
    sid_to_sess = {s.session_id: s for s in all_sessions}
    matched_sessions = []

    for rk in common_keys:
        matches = [sid for sid in sid_to_sess.keys() if sid in rk]
        if len(matches) == 1:
            sid = matches[0]
            s = sid_to_sess[sid]
            s.results_dict_key = rk
            r2s[rk] = sid
            if s not in matched_sessions:
                matched_sessions.append(s)
        elif len(matches) > 1:
            matches.sort(key=len, reverse=True)
            sid = matches[0]
            s = sid_to_sess[sid]
            s.results_dict_key = rk
            r2s[rk] = sid
            if s not in matched_sessions:
                matched_sessions.append(s)

    log.info("Mapped %d sessions to results.", len(matched_sessions))

    # 5. COMPUTE TIMELINE LENGTHS
    log.info("Step 5: Computing master timeline lengths...")
    expected_lengths = {}
    for s in matched_sessions:
        rk = s.results_dict_key
        n_frames = None
        if hasattr(s, "video_refs_df") and s.video_refs_df is not None and not s.video_refs_df.empty:
            if "role" in s.video_refs_df.columns and "n_frames" in s.video_refs_df.columns:
                primary = s.video_refs_df[s.video_refs_df["role"] == "primary"]
                if not primary.empty:
                    n_frames = int(primary.iloc[0]["n_frames"])
            if n_frames is None and "n_frames" in s.video_refs_df.columns:
                n_frames = int(s.video_refs_df["n_frames"].max())

        if hasattr(s, "photometry") and isinstance(s.photometry, dict) and "combined_filtered" in s.photometry:
            phot_len = len(s.photometry["combined_filtered"])
            n_frames = max(n_frames or 0, phot_len)

        if n_frames is None and rk in vfi:
            n_frames = int(vfi[rk].max()) + 1

        expected_lengths[rk] = n_frames

    # 6. WEAVE HELPER A RESULTS
    log.info("Step 6: Weaving results onto 80Hz timeline (Helper A)...")
    full_dfs = flatten_results_to_timeline(results, vfi, actual_lengths=expected_lengths)

    for s in matched_sessions:
        rk = s.results_dict_key
        if rk in full_dfs:
            s.syllables = full_dfs[rk]
            if "syllable" in s.syllables.columns:
                s.syllables["syllable"] = s.syllables["syllable"].astype("Int64")

    # 7. WEAVE HELPER B KINEMATICS & MERGE PHOTOMETRY
    log.info("Step 7: Densifying moseq_df and merging photometry (Helper B)...")
    ordered_moseq_dfs = build_aligned_dfs(matched_sessions, moseq_df, vfi, verbose=False)
    all_data_dfs = build_aligned_moseq_phot(matched_sessions, ordered_moseq_dfs)

    for i, s in enumerate(matched_sessions):
        s.all_data_df = all_data_dfs[i]
        if "syllable" in s.all_data_df.columns:
            s.all_data_df["syllable"] = s.all_data_df["syllable"].astype("Int64")

    # 8. SAVE OUTPUT PICKLES
    log.info("Step 8: Saving enriched sessions to %s...", SESSIONS_OUT_K240K)
    with open(SESSIONS_OUT_K240K, "wb") as f:
        pickle.dump(matched_sessions, f)
    log.info("Saved %s (%.2f GB).", SESSIONS_OUT_K240K.name, SESSIONS_OUT_K240K.stat().st_size / 1e9)

    log.info("Also updating main pickle at %s...", SESSIONS_OUT_MAIN)
    with open(SESSIONS_OUT_MAIN, "wb") as f:
        pickle.dump(matched_sessions, f)
    log.info("Saved %s (%.2f GB).", SESSIONS_OUT_MAIN.name, SESSIONS_OUT_MAIN.stat().st_size / 1e9)

    log.info("========================================================================")
    log.info(" Pipeline Finished Successfully in %.1f minutes!", (time.time() - t0_total) / 60.0)
    log.info("========================================================================")


if __name__ == "__main__":
    main()
