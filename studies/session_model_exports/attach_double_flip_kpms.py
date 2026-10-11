"""
attach_double_flip_kpms.py

Specialized attachment script for `flipped_double_project`.
This model contains 2x the results: 1 for original orientation, 1 for horizontal-flipped.
Because their VFIs are identical, we extract both into `pandas.DataFrame`s, rename the flipped
columns with a `_flipped` suffix, concatenate them horizontally, and append them via Helper B
to the original un-flipped `_sanitized.pkl` sessions.
"""

from __future__ import annotations

import pickle
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.simplefilter(action='ignore', category=FutureWarning)

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import keypoint_moseq as kpms

from syllable_generation.analysis.attach_kpms import _attach_kpms_full_to_sessions
from syllable_generation.analysis.replications.datta_utils import build_aligned_moseq_phot
from syllable_generation.analysis.utils.legacy_support import migrate_legacy_vfi
from syllable_generation.analysis.utils.sampling import (
    build_aligned_dfs,
)
from syllable_generation.analysis.utils.syll_results import (
    build_result_to_session_map,
    compute_actual_lengths_from_sessions,
    load_kpms_results_and_vfi,
)

PATH_PREFIX = "/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/"
DEFAULT_SESSIONS_IN = PATH_PREFIX + "data/jonathan_amph_sessions_sanitized.pkl"
DEFAULT_SESSIONS_OUT = PATH_PREFIX + "data/jonathan_amph_sessions_double_w_sylls.pkl"
DEFAULT_KPMS = PATH_PREFIX + "syllabizers/keypoint_moseq_syll/flipped_double_project/double_flip_activity_k100k/checkpoint.h5"


def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="Attach Double KPMS syllables to sessions.")
    parser.add_argument("--input", default=DEFAULT_SESSIONS_IN, help="Path to input sessions pkl")
    parser.add_argument("--output", default=DEFAULT_SESSIONS_OUT, help="Path to output sessions pkl")
    parser.add_argument("--kpms", default=DEFAULT_KPMS, help="Path to KPMS checkpoint.h5 or results.h5")
    parser.add_argument("--verbose", action="store_true", help="Enable detailed debug logging.")
    return parser.parse_args()


def main():
    args = get_args()

    print(f"Loading sessions from: {args.input}")
    sessions = pd.read_pickle(args.input)

    print(f"Loading KPMS results from {args.kpms}...")
    results, vfi = load_kpms_results_and_vfi(args.kpms, use_cpu=True)

    # --- Split Results into Normal / Flipped ---
    results_normal = {k: v for k, v in results.items() if not str(k).endswith('_flipped')}
    results_flipped = {k: v for k, v in results.items() if str(k).endswith('_flipped')}
    vfi_normal = {k: v for k, v in vfi.items() if not str(k).endswith('_flipped')}
    vfi_flipped = {k: v for k, v in vfi.items() if str(k).endswith('_flipped')}

    print(f"Split results into {len(results_normal)} standard and {len(results_flipped)} flipped items.")

    # --- Match keys using only normal ones (to establish expected lengths and order) ---
    # We enforce strict mapping here against our actual physical sessions.
    r2s = build_result_to_session_map(sessions, list(results_normal.keys()), strict=True, verbose=args.verbose)
    exp_len = compute_actual_lengths_from_sessions(sessions, r2s)

    # VFI Checks (using normal scale)
    vfi_normal = migrate_legacy_vfi(vfi_normal, exp_len, scale_factor=2.0, verbose=args.verbose)
    # the flipped VFIs are technically identical arrays, but we map them through migrate legacy just to be safe
    exp_len_flipped = {k + "_flipped": v for k, v in exp_len.items()}
    vfi_flipped = migrate_legacy_vfi(vfi_flipped, exp_len_flipped, scale_factor=2.0, verbose=args.verbose)

    # --- Flattening & Feature Extraction ---
    project_dir = Path(args.kpms).parents[1]
    model_name = Path(args.kpms).parent.name

    print("Computing moseq_df via standard library...")
    # WORKAROUND: KPMS has a bug where `compute_moseq_df` crashes if index.csv is missing
    # (UnboundLocalError) or if it's missing keys (IndexError).
    # We dynamically generate an index.csv covering all our keys (normal and flipped).
    index_csv = project_dir / "index.csv"
    pd.DataFrame({
        "name": list(results.keys()),
        "group": "default"
    }).to_csv(index_csv, index=False)

    # This generates df for EVERYTHING in results.h5 (normal and flipped combined natively)
    moseq_df = kpms.compute_moseq_df(str(project_dir), model_name, smooth_heading=True)

    # Split the DataFrame
    mask_flipped = moseq_df["name"].str.endswith("_flipped")
    moseq_df_normal = moseq_df[~mask_flipped].copy()
    moseq_df_flipped = moseq_df[mask_flipped].copy()

    # Strip '_flipped' from 'name' column purely so `build_aligned_dfs` matcher can find them
    # against `vfi_flipped` keys which we will ALSO temporarily strip.
    moseq_df_flipped["name_stripped"] = moseq_df_flipped["name"].str.replace("_flipped", "")
    moseq_df_flipped["name"] = moseq_df_flipped["name_stripped"]
    moseq_df_flipped.drop(columns=["name_stripped"], inplace=True)

    vfi_flipped_stripped = {k.replace("_flipped", ""): v for k, v in vfi_flipped.items()}

    print("Building fixed moseq dfs for Normal Set...")
    ordered_dfs_normal = build_aligned_dfs(sessions, moseq_df_normal, vfi_normal, verbose=args.verbose)

    print("Building fixed moseq dfs for Flipped Set...")
    # Passing vfi_flipped_stripped using the standard session mapping
    ordered_dfs_flipped = build_aligned_dfs(sessions, moseq_df_flipped, vfi_flipped_stripped, verbose=args.verbose)

    # --- Merge the Column Space ---
    print("Merging normal and flipped kinematics horizontally...")
    for f_df in ordered_dfs_flipped:
        if f_df is not None:
            # Rename columns to avoid dupes, but leave 'frame_index' / 'session_id' alone if present
            f_df.rename(columns=lambda c: f"{c}_flipped" if c not in ['frame_index', 'name'] else c, inplace=True)

    merged_dfs = []
    for i, (n_df, f_df) in enumerate(zip(ordered_dfs_normal, ordered_dfs_flipped)):
        if n_df is None and f_df is None:
            merged_dfs.append(None)
            continue

        if n_df is not None and f_df is not None:
             # they share the same physical frames, so horizontal concat is safe
             # drop overlap columns from flipped to be clean
             if 'frame_index' in f_df.columns:
                 f_df = f_df.drop(columns=['frame_index'])
             if 'name' in f_df.columns:
                 f_df = f_df.drop(columns=['name'])

             merged = pd.concat([n_df, f_df], axis=1)
             merged_dfs.append(merged)
        else:
            print(f"Warning: Session {i} has None df in one of the sets. Normal: {n_df is not None}, Flipped: {f_df is not None}")
            merged_dfs.append(None)

    # --- HELPER A Merge into Sessions Syllables DB ---
    # `merged_dfs` is identical in structure to standard flatten output natively,
    # except it is densely ordered matching `sessions`. We reconstruct a dict to pass.
    # We do a basic rebuild of `full_dfs` expected by `_attach_kpms_full_to_sessions`.
    # It expects: keys = rk, values = matching dataframe.
    full_dfs = {}
    r2s_keys = list(r2s.keys())
    for idx, (rk, si) in enumerate(r2s.items()):
        full_dfs[rk] = merged_dfs[si]

    print("Running Helper A (Attach)...")
    _attach_kpms_full_to_sessions(
        sessions,
        full_dfs,
        result_to_session=r2s,
        expected_lengths=exp_len
    )

    # --- HELPER B Merge into Photometry Timeline ---
    print("Running Helper B (Merge Photometry)...")
    all_data_dfs = build_aligned_moseq_phot(sessions, merged_dfs)

    for i, s in enumerate(sessions):
        s.all_data_df = all_data_dfs[i]

        rk = getattr(s, "results_dict_key", None)
        if rk and rk in vfi_normal and args.verbose:
                v_arr = vfi_normal[rk]
                v_max = v_arr.max()
                df_len = len(s.all_data_df)
                print(f"DEBUG FINAL SESS {i} ({s.session_id}): df len={df_len}, coverage={v_max / df_len:.4f}")

    print(f"Saving explicitly to {args.output}")
    with open(args.output, 'wb') as f:
        pickle.dump(sessions, f)

    print("Done.")

if __name__ == "__main__":
    main()
