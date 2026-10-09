#!/usr/bin/env python3
"""
extract_shmoseq_inputs.py
=============================================================================
Extract and prepare post-injection syllable streams and spatial tracking data
from all 30 sessions for exploratory shMoSeq modeling.

Methodology:
- Operates directly on the shared 0..99 Keypoint-MoSeq syllable codebook.
- Extracts continuous post-injection sequences without artificial downsampling
  or destructive chunking.
- Retains metadata (genotype, drug, animal_id, injection_frame) for post-hoc
  phenotyping.
=============================================================================
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd

# Paths
BASE_DIR = "/home/fsp585/Desktop/FiberphotometryCode"
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

RAW_DATA_PATH = os.environ.get(
    "SHMOSEQ_INPUT_PKL",
    os.path.join(BASE_DIR, "syllable_generation", "data", "jonathan_amph_sessions_run12_w_sylls.pkl")
)
OUTPUT_DIR = os.path.join(BASE_DIR, "syllable_generation", "analysis", "shmoseq", "output", "data")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "shmoseq_input_data.pkl")

def main():
    print("=" * 80)
    print("EXTRACTING SHMOSEQ INPUT DATA FROM 30 EXPERIMENTAL SESSIONS")
    print("=" * 80)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Loading raw sessions from: {RAW_DATA_PATH}")
    with open(RAW_DATA_PATH, "rb") as f:
        sessions = pickle.load(f)
    print(f"Loaded {len(sessions)} sessions.")
    
    # Load verified event frames table
    event_table_path = os.path.join(OUTPUT_DIR, "..", "tables", "session_event_frames_verified.csv")
    if os.path.exists(event_table_path):
        df_events = pd.read_csv(event_table_path)
        print(f"Loaded verified event frames from: {event_table_path}")
    else:
        df_events = None
        print("Warning: session_event_frames_verified.csv not found; falling back to metadata.")

    syllables_dict = {}
    pre_syllables_dict = {}
    centroids_dict = {}
    metadata_dict = {}
    
    unique_sylls_global = set()
    lengths = []
    
    for i, sess in enumerate(sessions):
        meta = sess.metadata
        aid = str(meta.get("animal_id", f"mouse_{i}"))
        genotype = meta.get("genotype", "WT")
        drug = meta.get("drug", "Saline")
        
        if df_events is not None and i < len(df_events):
            ev_row = df_events.iloc[i]
            nom_inj_f = int(ev_row["nominal_injection_frame"])
            det_rem_f = int(ev_row["detected_removal_frame"])
            det_ret_f = int(ev_row["detected_return_frame"])
            det_entry_f = int(ev_row["detected_entry_frame"])
            handling_dur_s = float(ev_row["handling_duration_sec"])
        else:
            nom_inj_f = int(meta.get("injection_frame", 0))
            det_rem_f = nom_inj_f
            det_ret_f = nom_inj_f
            det_entry_f = int(meta.get("entry_frame", 0))
            handling_dur_s = 0.0
        
        # Unique session key
        session_key = f"{aid}_{drug}"
        
        # Extract syllable series: STRICTLY starting at detected_return_frame
        # Excludes entire physical handling window from behavioral analysis!
        df_syll = sess.syllables
        post_df = df_syll.iloc[det_ret_f:] if det_ret_f < len(df_syll) else df_syll
        valid_mask = post_df["syllable"].notna()
        
        # Native 40 Hz syllable sequence (int32)
        syll_seq = post_df.loc[valid_mask, "syllable"].values.astype(np.int32)
        
        # Also extract pre-injection baseline (entry -> removal)
        pre_df = df_syll.iloc[det_entry_f:det_rem_f] if det_rem_f > det_entry_f else df_syll.iloc[:nom_inj_f]
        valid_pre = pre_df["syllable"].notna()
        pre_syll_seq = pre_df.loc[valid_pre, "syllable"].values.astype(np.int32)
        pre_syllables_dict[session_key] = pre_syll_seq
        
        # Centroid coordinates (float32)
        if "centroid_0" in post_df.columns and "centroid_1" in post_df.columns:
            centroids = post_df.loc[valid_mask, ["centroid_0", "centroid_1"]].values.astype(np.float32)
        else:
            centroids = np.zeros((len(syll_seq), 2), dtype=np.float32)
            
        syllables_dict[session_key] = syll_seq
        centroids_dict[session_key] = centroids
        
        metadata_dict[session_key] = {
            "session_idx": i,
            "animal_id": aid,
            "genotype": genotype,
            "drug": drug,
            "cohort": f"{genotype}_{drug}",
            "nominal_injection_frame": nom_inj_f,
            "detected_removal_frame": det_rem_f,
            "detected_return_frame": det_ret_f,
            "detected_entry_frame": det_entry_f,
            "handling_duration_sec": handling_dur_s,
            "num_frames_40hz": len(syll_seq),
            "duration_minutes": len(syll_seq) / 40.0125 / 60.0,
            "time_course_reference": "time_since_return",
            "original_session_id": getattr(sess, "session_id", f"sess_{i}"),
        }
        
        lengths.append(len(syll_seq))
        for s in np.unique(syll_seq):
            unique_sylls_global.add(int(s))
            
    print(f"\nExtraction Summary (Strict Post-Return Alignment):")
    print(f"  - Total sessions extracted: {len(syllables_dict)}")
    print(f"  - Total unique syllables observed: {len(unique_sylls_global)} (range {min(unique_sylls_global)}..{max(unique_sylls_global)})")
    print(f"  - Post-injection sequence length (40 Hz frames): min={min(lengths)}, max={max(lengths)}, median={np.median(lengths):.1f}")
    print(f"  - Equivalent duration at 40.0125 Hz: min={min(lengths)/40.0125/60:.1f}m, max={max(lengths)/40.0125/60:.1f}m, median={np.median(lengths)/40.0125/60:.1f}m")
    
    # Cohort breakdown
    cohort_counts = {}
    for k, m in metadata_dict.items():
        c = m["cohort"]
        cohort_counts[c] = cohort_counts.get(c, 0) + 1
    print(f"  - Cohort breakdown: {cohort_counts}")
    
    # Pack into output
    bundle = {
        "syllables_dict": syllables_dict,
        "pre_syllables_dict": pre_syllables_dict,
        "centroids_dict": centroids_dict,
        "metadata_dict": metadata_dict,
        "n_syllables": max(unique_sylls_global) + 1,
        "fps": 40.0125,  # Native video/syllable rate (clock timeline is 80 Hz with step 2)
    }
    
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(bundle, f)
        
    print(f"\nSuccessfully saved shMoSeq input bundle to:\n  {OUTPUT_FILE}")
    print("=" * 80)

if __name__ == "__main__":
    main()
