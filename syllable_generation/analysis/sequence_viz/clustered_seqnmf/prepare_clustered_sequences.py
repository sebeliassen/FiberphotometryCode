#!/usr/bin/env python3
"""
prepare_clustered_sequences.py
=============================================================================
Maps the 100 Keypoint-MoSeq syllables into 14 functional Meta-Syllables based
on the 54 expert human annotations (covering 92.7% of bouts and 90.5% of duration).

Generates two continuous sequence streams across the cohort using fast vectorization:
1. Mode 1: Run-Length Compressed (Semi-Markov macro-bouts, deduplicated)
2. Mode 2: Bout-Preserving (preserving individual bout steps)

Computes the 14x14 empirical macro-transition matrix and saves the unified bundle.
=============================================================================
"""

import os
import sys
import time
import pickle
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path("/home/fsp585/Desktop/FiberphotometryCode")
DATA_DIR = REPO_ROOT / "syllable_generation/analysis/sequence_viz/data"
OUT_DIR = REPO_ROOT / "syllable_generation/analysis/sequence_viz/clustered_seqnmf"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Define the 13 ground-truth functional meta-syllables + 1 tail class
META_SYLLABLE_DEFS = {
    0: {"name": "Walk_Right", "label": "M01: Walk Right", "sylls": [1, 11]},
    1: {"name": "Walk_Left", "label": "M02: Walk Left", "sylls": [3, 23, 55]},
    2: {"name": "Walk_Forward", "label": "M03: Walk Forward", "sylls": [19, 38]},
    3: {"name": "Sprint_Run", "label": "M04: Sprint / Dash", "sylls": [0, 13, 17, 21, 26, 33, 41]},
    4: {"name": "Turn_Right", "label": "M05: Turn Right", "sylls": [2, 30]},
    5: {"name": "Turn_Left", "label": "M06: Turn Left", "sylls": [6, 29]},
    6: {"name": "Rear_Elevate", "label": "M07: Rear / Wall Lean", "sylls": [12, 14, 16, 31]},
    7: {"name": "Touchdown_Descent", "label": "M08: Postural Descent", "sylls": [4, 7, 37]},
    8: {"name": "Sniff_Investigate", "label": "M09: Sniff / Investigate", "sylls": [8, 9, 10, 15, 18, 20, 24, 25, 36, 39]},
    9: {"name": "Look_Orient", "label": "M10: Look / Orient", "sylls": [5, 22, 28, 34, 40, 42]},
    10: {"name": "Groom_Module", "label": "M11: Grooming", "sylls": [27, 44, 52, 58, 60, 61, 63, 65, 66]},
    11: {"name": "Stationary_Rest", "label": "M12: Stationary Rest", "sylls": [32, 43, 49]},
    12: {"name": "Tracker_Artifact", "label": "M13: Tracker Artifact", "sylls": [35]},
    13: {"name": "Tail_Unlabeled", "label": "M14: Unlabeled Tail", "sylls": []},
}

# Create fast lookup array (syllable_id -> meta_id)
SYLL_TO_META_ARR = np.full(100, 13, dtype=int)
for meta_id, meta in META_SYLLABLE_DEFS.items():
    for s in meta["sylls"]:
        SYLL_TO_META_ARR[s] = meta_id

for s in range(100):
    if s not in [item for sublist in [m["sylls"] for m in META_SYLLABLE_DEFS.values()] for item in sublist]:
        META_SYLLABLE_DEFS[13]["sylls"].append(s)

META_NAMES = [META_SYLLABLE_DEFS[i]["name"] for i in range(len(META_SYLLABLE_DEFS))]


def main():
    t0 = time.time()
    print(f"Meta-Syllable mapping defined across {len(META_SYLLABLE_DEFS)} classes.")
    for meta_id, meta in META_SYLLABLE_DEFS.items():
        print(f"  {meta['label']:25s} : {len(meta['sylls']):2d} syllables")

    # Load unified sequence dataset
    src_pkl = DATA_DIR / "unified_sequence_dataset.pkl"
    with open(src_pkl, "rb") as f:
        bundle = pickle.load(f)

    sessions = bundle["sessions"]
    print(f"\nLoaded unified dataset with {len(sessions)} sessions.")

    processed_sessions = []
    all_meta_bouts_mode1 = []
    all_meta_bouts_mode2 = []
    total_raw_bouts = 0

    for sess in sessions:
        df = sess["bouts_df"]
        n_bouts = len(df)
        total_raw_bouts += n_bouts
        
        sylls = df["syllable"].values
        durs = df["duration_sec"].values
        starts = df["start_frame"].values
        ends = df["end_frame"].values
        
        meta_arr = SYLL_TO_META_ARR[sylls]
        all_meta_bouts_mode2.extend(meta_arr)
        
        if n_bouts > 0:
            change_pts = np.where(meta_arr[:-1] != meta_arr[1:])[0] + 1
            g_starts = np.concatenate(([0], change_pts))
            g_ends = np.concatenate((change_pts, [n_bouts]))
            
            rle_meta = meta_arr[g_starts]
            rle_start_frames = starts[g_starts]
            rle_end_frames = ends[g_ends - 1]
            
            cum_durs = np.concatenate(([0.0], np.cumsum(durs)))
            rle_durs = cum_durs[g_ends] - cum_durs[g_starts]
            rle_sub_counts = g_ends - g_starts
            
            all_meta_bouts_mode1.extend(rle_meta)
            
            macro_df = pd.DataFrame({
                "macro_bout_idx": np.arange(len(rle_meta)),
                "meta_syllable": rle_meta,
                "meta_label": [META_NAMES[m] for m in rle_meta],
                "duration_sec": rle_durs,
                "start_frame": rle_start_frames,
                "end_frame": rle_end_frames,
                "num_sub_bouts": rle_sub_counts,
            })
        else:
            macro_df = pd.DataFrame()
            
        processed_sessions.append({
            "session_idx": sess["session_idx"],
            "animal_id": sess["animal_id"],
            "genotype": sess["genotype"],
            "drug": sess["drug"],
            "raw_meta_syllables": meta_arr,
            "macro_bouts_df": macro_df,
        })

    print(f"\nVectorized Sequence Transformation Complete in {time.time() - t0:.2f}s:")
    print(f"  Total Raw Micro-Bouts: {total_raw_bouts:,}")
    print(f"  Mode 1 Macro-Bouts (Deduplicated): {len(all_meta_bouts_mode1):,} (Compression: {len(all_meta_bouts_mode1)/total_raw_bouts*100:.1f}%)")
    print(f"  Mode 2 Bouts (Preserved): {len(all_meta_bouts_mode2):,}")

    # Compute 14x14 Macro-Transition Probability Matrix (Mode 1 - Deduplicated)
    num_meta = len(META_SYLLABLE_DEFS)
    trans_counts = np.zeros((num_meta, num_meta), dtype=int)
    for sess in processed_sessions:
        seq = sess["macro_bouts_df"]["meta_syllable"].values
        if len(seq) > 1:
            np.add.at(trans_counts, (seq[:-1], seq[1:]), 1)

    row_sums = trans_counts.sum(axis=1, keepdims=True)
    trans_prob_mode1 = np.divide(trans_counts, row_sums, out=np.zeros_like(trans_counts, dtype=float), where=row_sums > 0)

    trans_df = pd.DataFrame(trans_prob_mode1, index=META_NAMES, columns=META_NAMES)
    trans_csv = OUT_DIR / "macro_syllable_transition_matrix_mode1.csv"
    trans_df.to_csv(trans_csv)
    print(f"✓ Saved 14x14 Macro-Transition Matrix to {trans_csv}")

    clustered_bundle = {
        "meta_syllable_defs": META_SYLLABLE_DEFS,
        "syll_to_meta_arr": SYLL_TO_META_ARR,
        "sessions": processed_sessions,
        "macro_transition_matrix_mode1": trans_prob_mode1,
        "meta_names": META_NAMES,
    }
    bundle_file = OUT_DIR / "clustered_sequence_dataset.pkl"
    with open(bundle_file, "wb") as f:
        pickle.dump(clustered_bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\n✓ Saved complete clustered dataset bundle to {bundle_file} ({os.path.getsize(bundle_file)/1e6:.2f} MB)")
    print(f"Total time elapsed: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
