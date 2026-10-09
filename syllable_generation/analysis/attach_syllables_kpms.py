"""
attach_syllables_kpms.py

Logic:
1. Load Data
2. Run "Helper A" (flatten & attach syllables)
3. SHIM: Add 'frame_index' column to the attached dataframes
4. Run "Helper B" (merge photometry)
5. Save
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

np.bool8 = np.bool_
import warnings

# Silence FutureWarnings (e.g. from Pandas ffill)
warnings.simplefilter(action='ignore', category=FutureWarning)

# === REPO ROOT BOILERPLATE ===
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# === IMPORTS ===
# Helper A tools
# Helper B tools
import keypoint_moseq as kpms

from syllable_generation.analysis.attach_kpms import _attach_kpms_full_to_sessions
from syllable_generation.analysis.replications.datta_utils import build_aligned_moseq_phot
from syllable_generation.analysis.utils.legacy_support import migrate_legacy_vfi
from syllable_generation.analysis.utils.sampling import (
    build_aligned_dfs,
    flatten_results_to_timeline,
)
from syllable_generation.analysis.utils.syll_results import (
    build_result_to_session_map,
    compute_actual_lengths_from_sessions,
    load_kpms_results_and_vfi,
)

# === CONSTANTS ===
PATH_PREFIX = "/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/"
# === CONSTANTS (Defaults) ===
DEFAULT_SESSIONS_IN = PATH_PREFIX + "data/jonathan_ifdn_sessions_sanitized.pkl"
DEFAULT_SESSIONS_OUT = PATH_PREFIX + "data/jonathan_ifdn_sessions_w_sylls.pkl"
DEFAULT_KPMS = PATH_PREFIX + "syllabizers/keypoint_moseq_syll/demo_project/full-run/checkpoint.h5"

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="Attach KPMS syllables to sessions.")
    parser.add_argument("--input", default=DEFAULT_SESSIONS_IN, help="Path to input sessions pkl")
    parser.add_argument("--output", default=DEFAULT_SESSIONS_OUT, help="Path to output sessions pkl")
    parser.add_argument("--kpms", default=DEFAULT_KPMS, help="Path to KPMS checkpoint.h5 or results.h5")
    parser.add_argument("--vfi", default=None, help="Explicit path to video_frame_indexes.npz")
    parser.add_argument("--legacy-rate", type=float, default=2.0,
                        help="Factor to scale Legacy VFIs by if detected (default 2.0).")
    parser.add_argument("--results-upsample-rate", type=int, default=1,
                        help="Factor to tile/upsample results arrays (e.g. 2). Default 1 (no change).")
    parser.add_argument("--first-n-sessions", type=int, default=None,
                        help="Only process the first N sessions (for faster debugging).")
    parser.add_argument("--verbose", action="store_true", help="Enable detailed debug logging.")
    return parser.parse_args()


def main():
    args = get_args()

    # --- STEP 1: LOAD ---
    print(f"Loading sessions from: {args.input}")
    try:
        sessions = pd.read_pickle(args.input)
    except FileNotFoundError:
        print(f"Error: Input file not found: {args.input}")
        return

    if args.first_n_sessions is not None:
        if args.verbose:
            print(f"DEBUG: Truncating sessions to first {args.first_n_sessions}")
        sessions = sessions[:args.first_n_sessions]
        # Dynamically inject suffix into output path
        out_path = Path(args.output)
        args.output = str(out_path.parent / f"{out_path.stem}_n{args.first_n_sessions}{out_path.suffix}")

    for i, s in enumerate(sessions):
        if args.verbose:
            if hasattr(s, 'photometry') and 'combined_filtered' in s.photometry:
                print(f"DEBUG LOAD SESS {i} ({s.session_id}): initial PhotLen = {len(s.photometry['combined_filtered'])}")

    print(f"Working with {len(sessions)} sessions. Final output will be: {args.output}")

    print(f"Loading KPMS results from {args.kpms}...")
    # Determine which VFI path to use
    if args.vfi:
        vfi_path = Path(args.vfi)
        results, _ = load_kpms_results_and_vfi(args.kpms, use_cpu=True)
        print(f"Loading explicit VFI from {vfi_path}...")
        with np.load(vfi_path) as z:
            vfi = {k: z[k].astype(int) for k in z.files}
    else:
        results, vfi = load_kpms_results_and_vfi(args.kpms, use_cpu=True)

    # --- SUBSET FILTERING (for debugging) ---
    if args.first_n_sessions is not None:
        kept_sids = [s.session_id for s in sessions]
        results = {k: v for k, v in results.items() if any(sid in k for sid in kept_sids)}
        vfi = {k: v for k, v in vfi.items() if k in results}
        if args.verbose:
            print(f"DEBUG: Filtered results to {len(results)} keys matching the {len(sessions)} sessions.")

    # Mapping must happen early so sessions get 'results_dict_key' for rate detection
    r2s = build_result_to_session_map(sessions, list(results.keys()), strict=True, verbose=args.verbose)

    # Filter sessions to only those that actually have a matching key in results
    matched_sessions = [s for s in sessions if getattr(s, "results_dict_key", "")]
    if len(matched_sessions) < len(sessions):
        print(f"Filtering: Keeping {len(matched_sessions)} matched sessions out of {len(sessions)} total.")
        sessions = matched_sessions
        # Re-build mapping with the filtered sessions to ensure correct indices (0 to N-1)
        r2s = build_result_to_session_map(sessions, list(results.keys()), strict=True, verbose=args.verbose)

    exp_len = compute_actual_lengths_from_sessions(sessions, r2s)

    # --- VFI MIGRATION / SCALING CHECK ---
    print("Checking VFI for legacy downsampled indices...")
    vfi = migrate_legacy_vfi(vfi, exp_len, scale_factor=args.legacy_rate if hasattr(args, 'legacy_rate') else 2.0, verbose=args.verbose)

    print("Flattening results...")
    full_dfs = flatten_results_to_timeline(results, vfi, actual_lengths=exp_len)

    # --- STEP 2: KPMS COMPUTE (For Kinematics/all_data_df) ---
    print("Computing moseq_df via standard library...")
    project_dir = Path(args.kpms).parents[1]
    model_name = Path(args.kpms).parent.name
    moseq_df = kpms.compute_moseq_df(str(project_dir), model_name, smooth_heading=True)

    # Add group metadata
    index_csv = project_dir / "index.csv"
    if index_csv.exists():
        groups_df = pd.read_csv(index_csv)
        if "name" in groups_df.columns and "group" in groups_df.columns:
            name_to_group = dict(zip(groups_df["name"], groups_df["group"]))
            moseq_df["group"] = moseq_df["name"].map(name_to_group)
            print("Added group metadata from index.csv")

    # Add velocity, angular_velocity, onset if missing (standard compute should have them)
    if "onset" not in moseq_df.columns and "syllable" in moseq_df.columns:
        moseq_df["onset"] = moseq_df.groupby("name")["syllable"].transform(lambda x: x.diff() != 0)
        moseq_df.loc[moseq_df.groupby("name").head(1).index, "onset"] = True

    # --- STEP 3: HELPER A LOGIC (Attach Syllables) ---
    # This attaches the dataframe to s.syllables['kpms:UUID']
    print("Running Helper A (Attach)...")
    _attach_kpms_full_to_sessions(
        sessions,
        full_dfs,
        result_to_session=r2s,
        expected_lengths=exp_len
    )

    # --- STEP 4: HELPER B LOGIC (Merge Photometry) ---
    print("Building fixed moseq dfs for Helper B (Upsampling & Densifying)...")
    # Use the new utility logic
    ordered_dfs = build_aligned_dfs(
        sessions,
        moseq_df,
        vfi,
        verbose=args.verbose
    )

    print("Running Helper B (Merge Photometry)...")
    all_data_dfs = build_aligned_moseq_phot(sessions, ordered_dfs)

    for i, s in enumerate(sessions):
        s.all_data_df = all_data_dfs[i]

        # Debug: Detailed cardinality check
        if args.verbose:
            rk = getattr(s, "results_dict_key", None)
            if rk and rk in vfi:
                v_arr = vfi[rk]
                v_len = len(v_arr)
                v_max = v_arr.max()
                v_unique = len(np.unique(v_arr))
                df_len = len(s.all_data_df)
                print(f"DEBUG FINAL SESS {i} ({s.session_id}):")
                print(f"  > all_data_df Len: {df_len}")
                print(f"  > VFI Len:        {v_len}")
                print(f"  > VFI Max Index:  {v_max}")
                print(f"  > VFI Unique:     {v_unique}")
                print(f"  > Delta (Phot-VFI): {df_len - v_unique}")
                print(f"  > Coverage Ratio: {v_max / df_len:.4f}")

        # Optional: ensure syllable is Int64 (legacy consistency)
        # If s.syllables is a DataFrame (single_df=True):
        if hasattr(s.syllables, "columns"):
             if "syllable" in s.syllables.columns:
                 s.syllables["syllable"] = s.syllables["syllable"].astype("Int64")
        # If s.syllables is a dict (single_df=False):
        elif isinstance(s.syllables, dict):
            for k, df in s.syllables.items():
                if "syllable" in df.columns:
                    df["syllable"] = df["syllable"].astype("Int64")

    # --- STEP 5: SAVE ---
    print(f"Saving recreated sessions to {args.output}")
    with open(args.output, 'wb') as f:
        pickle.dump(sessions, f)

    print("Done.")

if __name__ == "__main__":
    main()
