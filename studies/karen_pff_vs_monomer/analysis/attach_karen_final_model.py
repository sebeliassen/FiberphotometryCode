#!/usr/bin/env python3
"""
attach_karen_final_model.py

Pipeline to:
1. Reindex syllables & extract results.h5 from karen_project_2/<model_name>/checkpoint.h5
2. Compute moseq_df with kinematics (velocity, heading, angular velocity, onset)
3. Map recording keys to session objects in sessions_karen_project_dirty.pkl
4. Weave 40Hz model results onto the 80Hz master recording timeline using video_frame_indexes.npz
5. Merge photometry channels (or placeholder NaNs if absent)
6. Attach s.all_data_df and s.syllables to each session and save the enriched pickle
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Force CPU execution for JAX / KPMS extraction to avoid VRAM contention
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

# Ensure repo root is on sys.path
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
log = logging.getLogger("attach_karen")

PROJECT_DIR = REPO_ROOT / "syllable_generation" / "syllabizers" / "keypoint_moseq_syll" / "karen_project_2"
DEFAULT_MODEL = "final_karen_model_k240k"
DEFAULT_INPUT = REPO_ROOT / "syllable_generation" / "data" / "sessions_karen_project_dirty.pkl"
DEFAULT_OUTPUT = REPO_ROOT / "syllable_generation" / "data" / "sessions_karen_project_w_sylls_k240k.pkl"


def parse_args():
    parser = argparse.ArgumentParser(description="Extract and attach KPMS model syllables to Karen project sessions.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL, help=f"Model directory name in karen_project_2 (default: {DEFAULT_MODEL})")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Path to input dirty sessions pkl")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Path to output enriched sessions pkl")
    parser.add_argument("--update-main-pkl", action="store_true", help="Also overwrite sessions_karen_project_w_sylls.pkl")
    return parser.parse_args()


def main():
    args = parse_args()
    model_name = args.model_name
    model_dir = PROJECT_DIR / model_name
    results_path = model_dir / "results.h5"
    vfi_path = model_dir / "video_frame_indexes.npz"
    sessions_in = Path(args.input)
    sessions_out = Path(args.output)

    log.info("========================================================================")
    log.info("=== Karen Syllable Extraction & Session Enrichment ===")
    log.info("Project Dir:  %s", PROJECT_DIR)
    log.info("Model Name:   %s", model_name)
    log.info("Input PKL:    %s", sessions_in)
    log.info("Output PKL:   %s", sessions_out)
    log.info("========================================================================")

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    # --- STEP 1: LOAD OR EXTRACT RESULTS.H5 ---
    if results_path.exists():
        log.info("Found existing results.h5 at %s. Loading results...", results_path)
        results = kpms.load_results(str(PROJECT_DIR), model_name)
    else:
        log.info("results.h5 not found. Reindexing checkpoint and extracting results...")
        try:
            kpms.reindex_syllables_in_checkpoint(str(PROJECT_DIR), model_name)
            log.info("Reindexed syllables in checkpoint successfully.")
        except Exception as e:
            log.warning("reindex_syllables_in_checkpoint encountered: %s", e)

        log.info("Loading checkpoint from %s...", model_dir)
        model, data, metadata, current_iter = kpms.load_checkpoint(str(PROJECT_DIR), model_name)
        log.info("Loaded checkpoint at iteration %s. Extracting results to %s...", current_iter, results_path)
        results = kpms.extract_results(model, metadata, str(PROJECT_DIR), model_name, save_results=True)
        log.info("Saved results.h5 successfully (%d recording parts).", len(results))

        # Free checkpoint model memory
        del model, data, metadata
        gc.collect()

    # --- STEP 2: LOAD VFI ---
    if not vfi_path.exists():
        raise FileNotFoundError(f"Missing VFI file: {vfi_path}")
    log.info("Loading video frame indexes from %s...", vfi_path)
    with np.load(vfi_path) as z:
        vfi = {k: z[k].astype(int) for k in z.files}
    log.info("Loaded VFI for %d recordings.", len(vfi))

    # --- STEP 3: COMPUTE MOSEQ DF (Kinematics) ---
    # Ensure index.csv exists to avoid kpms compute_moseq_df bug
    index_csv = PROJECT_DIR / "index.csv"
    if not index_csv.exists():
        log.info("Creating %s for project...", index_csv)
        df_index = pd.DataFrame({"name": list(results.keys()), "group": ["default"] * len(results)})
        df_index.to_csv(index_csv, index=False)

    log.info("Computing moseq_df via keypoint_moseq...")
    moseq_df = kpms.compute_moseq_df(str(PROJECT_DIR), model_name, smooth_heading=True)
    log.info("moseq_df computed: %d rows, %d columns.", len(moseq_df), len(moseq_df.columns))

    # Ensure onset column exists
    if "onset" not in moseq_df.columns and "syllable" in moseq_df.columns:
        moseq_df["onset"] = moseq_df.groupby("name")["syllable"].transform(lambda x: x.diff() != 0)
        moseq_df.loc[moseq_df.groupby("name").head(1).index, "onset"] = True

    # Map groups if index.csv has meaningful groups
    if index_csv.exists():
        groups_df = pd.read_csv(index_csv)
        if "name" in groups_df.columns and "group" in groups_df.columns:
            name_to_group = dict(zip(groups_df["name"], groups_df["group"]))
            moseq_df["group"] = moseq_df["name"].map(name_to_group)

    # Filter results and VFI to common keys
    common_keys = sorted(list(set(results.keys()) & set(vfi.keys())))
    log.info("Filtering to %d common keys present in both results and VFI.", len(common_keys))
    results = {k: results[k] for k in common_keys}
    vfi = {k: vfi[k] for k in common_keys}
    moseq_df = moseq_df[moseq_df["name"].isin(common_keys)].copy()

    # --- STEP 4: LOAD SESSIONS AND MAP TO RESULTS KEYS ---
    log.info("Loading input sessions from %s...", sessions_in)
    with open(sessions_in, "rb") as f:
        all_sessions = pickle.load(f)
    log.info("Loaded %d total sessions from pickle.", len(all_sessions))

    # Build direct session mapping by session_id substring match in result key
    r2s = {}
    sid_to_sess = {s.session_id: s for s in all_sessions}
    matched_sessions = []

    for rk in common_keys:
        # Match session_id contained within recording key
        matches = [sid for sid in sid_to_sess.keys() if sid in rk]
        if len(matches) == 1:
            sid = matches[0]
            s = sid_to_sess[sid]
            s.results_dict_key = rk
            r2s[rk] = sid
            if s not in matched_sessions:
                matched_sessions.append(s)
        elif len(matches) == 0:
            log.warning("No session match found for recording key: %s", rk)
        else:
            # Tie breaker: longest matching session ID
            matches.sort(key=len, reverse=True)
            sid = matches[0]
            s = sid_to_sess[sid]
            s.results_dict_key = rk
            r2s[rk] = sid
            if s not in matched_sessions:
                matched_sessions.append(s)

    log.info("Matched %d sessions out of %d active VFI keys.", len(matched_sessions), len(common_keys))

    # --- STEP 5: COMPUTE EXPECTED FULL-SESSION TIMELINE LENGTHS ---
    expected_lengths = {}
    for s in matched_sessions:
        rk = s.results_dict_key
        # Primary reference: n_frames in video_refs_df
        n_frames = None
        if hasattr(s, "video_refs_df") and s.video_refs_df is not None and not s.video_refs_df.empty:
            if "role" in s.video_refs_df.columns and "n_frames" in s.video_refs_df.columns:
                primary = s.video_refs_df[s.video_refs_df["role"] == "primary"]
                if not primary.empty:
                    n_frames = int(primary.iloc[0]["n_frames"])
            if n_frames is None and "n_frames" in s.video_refs_df.columns:
                n_frames = int(s.video_refs_df["n_frames"].max())

        # Fallback to photometry length if available
        if hasattr(s, "photometry") and isinstance(s.photometry, dict) and "combined_filtered" in s.photometry:
            phot_len = len(s.photometry["combined_filtered"])
            n_frames = max(n_frames or 0, phot_len)

        # Fallback to VFI max index + 1
        if n_frames is None and rk in vfi:
            n_frames = int(vfi[rk].max()) + 1

        expected_lengths[rk] = n_frames

    log.info("Computed expected timeline lengths (Min=%d, Median=%d, Max=%d).",
             min(expected_lengths.values()), int(np.median(list(expected_lengths.values()))), max(expected_lengths.values()))

    # --- STEP 6: WEAVE HELPER A RESULTS (s.syllables) ---
    log.info("Weaving model results onto 80Hz timeline (Helper A)...")
    full_dfs = flatten_results_to_timeline(results, vfi, actual_lengths=expected_lengths)

    for s in matched_sessions:
        rk = s.results_dict_key
        if rk in full_dfs:
            s.syllables = full_dfs[rk]
            # Ensure nullable Int64 dtype
            if "syllable" in s.syllables.columns:
                s.syllables["syllable"] = s.syllables["syllable"].astype("Int64")

    # --- STEP 7: WEAVE HELPER B KINEMATICS & MERGE PHOTOMETRY (s.all_data_df) ---
    log.info("Densifying and aligning moseq_df to 80Hz timeline (Helper B)...")
    ordered_moseq_dfs = build_aligned_dfs(matched_sessions, moseq_df, vfi, verbose=False)

    log.info("Merging photometry channels / placeholders into all_data_df...")
    all_data_dfs = build_aligned_moseq_phot(matched_sessions, ordered_moseq_dfs)

    for i, s in enumerate(matched_sessions):
        s.all_data_df = all_data_dfs[i]
        if "syllable" in s.all_data_df.columns:
            s.all_data_df["syllable"] = s.all_data_df["syllable"].astype("Int64")

    # --- STEP 8: SAVE ENRICHED SESSIONS ---
    sessions_out.parent.mkdir(parents=True, exist_ok=True)
    log.info("Saving %d enriched sessions to %s...", len(matched_sessions), sessions_out)
    with open(sessions_out, "wb") as f:
        pickle.dump(matched_sessions, f)

    file_size_gb = sessions_out.stat().st_size / 1e9
    log.info("Successfully created %s (%.2f GB, %d sessions).", sessions_out.name, file_size_gb, len(matched_sessions))

    if args.update_main_pkl:
        main_out = REPO_ROOT / "syllable_generation" / "data" / "sessions_karen_project_w_sylls.pkl"
        log.info("Also updating main pickle at %s...", main_out)
        with open(main_out, "wb") as f:
            pickle.dump(matched_sessions, f)
        log.info("Updated main pickle successfully.")

    log.info("=== Syllable Enrichment Pipeline Completed Successfully ===")


if __name__ == "__main__":
    main()
