#!/usr/bin/env python3
"""
extract_shmoseq_inputs_8kp.py
=============================================================================
Extract and prepare post-injection syllable streams from 8KP Selected Candidate
(amph_8kp_k150k_run_03) across all 30 experimental sessions for shMoSeq modeling.

Methodology:
- Extracts directly from amph_8kp_k150k_run_03/results.h5.
- Uses verified physical event frames from session_event_frames_verified.csv.
- Strictly aligns post-injection syllable sequence starting at detected_return_frame
  (converted from 80Hz camera units to 40Hz model units), completely excluding the
  physical handling / injection window.
- Saves bundle to output_8kp_run03/data/shmoseq_input_data_8kp.pkl to maintain
  clean separation from the 17KP baseline.
=============================================================================
"""

import pickle
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

REPO_ROOT = Path("/home/fsp585/Desktop/FiberphotometryCode")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RESULTS_PATH = REPO_ROOT / "syllable_generation" / "syllabizers" / "keypoint_moseq_syll" / "amph_project_8kp" / "amph_8kp_k150k_run_03" / "results.h5"
EVENTS_CSV = REPO_ROOT / "syllable_generation" / "analysis" / "shmoseq" / "output" / "tables" / "session_event_frames_verified.csv"
OUTPUT_DIR = REPO_ROOT / "syllable_generation" / "analysis" / "shmoseq" / "output_8kp_run03" / "data"
OUTPUT_FILE = OUTPUT_DIR / "shmoseq_input_data_8kp.pkl"


def main():
    print("=" * 80)
    print("EXTRACTING 8KP (RUN 03) SHMOSEQ INPUT DATA FROM 30 EXPERIMENTAL SESSIONS")
    print(f"Results H5: {RESULTS_PATH}")
    print(f"Events CSV: {EVENTS_CSV}")
    print(f"Output:     {OUTPUT_FILE}")
    print("=" * 80, flush=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df_events = pd.read_csv(EVENTS_CSV)
    print(f"Loaded {len(df_events)} verified session event records.")

    # Load original session metadata for exact chamber/day/trial mapping
    sessions_pkl_path = REPO_ROOT / "syllable_generation" / "data" / "jonathan_amph_sessions_run12_w_sylls.pkl"
    with open(sessions_pkl_path, "rb") as f_s:
        raw_sessions = pickle.load(f_s)

    sess_meta_map = {}
    for s in raw_sessions:
        m = s.metadata
        s_aid = str(m["animal_id"])
        s_drug = str(m["drug"])
        sess_meta_map[(s_aid, s_drug)] = m

    syllables_dict = {}
    pre_syllables_dict = {}
    centroids_dict = {}
    metadata_dict = {}
    unique_sylls = set()
    lengths = []

    with h5py.File(RESULTS_PATH, "r") as f_res:
        res_keys = list(f_res.keys())
        print(f"Loaded {len(res_keys)} sessions from results.h5.")

        for idx, ev_row in df_events.iterrows():
            aid = str(ev_row["animal_id"])
            geno = str(ev_row["genotype"])
            drug = str(ev_row["drug"])
            session_key = f"{aid}_{drug}"

            # Convert 80Hz camera frame indices to 40Hz model frame indices
            det_entry_40 = int(ev_row["detected_entry_frame"]) // 2
            det_rem_40 = int(ev_row["detected_removal_frame"]) // 2
            det_ret_40 = int(ev_row["detected_return_frame"]) // 2
            nom_inj_40 = int(ev_row["nominal_injection_frame"]) // 2

            # Find exact matching key in results.h5 using day, chamber, and trial/timestamp
            s_m = sess_meta_map[(aid, drug)]
            day_str = "day1" if "sal" in drug.lower() else "day2"
            chamber = s_m["chamber"]
            trial = s_m["trial_id"]
            ts = s_m["timestamp"]
            matching = [k for k in res_keys if day_str in k and chamber in k and (ts in k or trial in k)]
            if len(matching) != 1:
                raise KeyError(f"Failed unique match for animal {aid}, drug {drug}: found {matching}")
            matched_key = matching[0]

            full_sylls = f_res[matched_key]["syllable"][:]
            full_centroid = f_res[matched_key]["centroid"][:]

            # Post-injection stream: strictly after detected animal return
            post_sylls = full_sylls[det_ret_40:]
            post_centroids = full_centroid[det_ret_40:]

            # Mask out any trailing unsegmented frames (< 0)
            valid_post = post_sylls >= 0
            syll_seq = post_sylls[valid_post].astype(np.int32)
            cent_seq = post_centroids[valid_post].astype(np.float32)

            # Pre-injection stream: entry to removal
            pre_sylls = full_sylls[det_entry_40:det_rem_40]
            valid_pre = pre_sylls >= 0
            pre_syll_seq = pre_sylls[valid_pre].astype(np.int32)

            syllables_dict[session_key] = syll_seq
            pre_syllables_dict[session_key] = pre_syll_seq
            centroids_dict[session_key] = cent_seq

            metadata_dict[session_key] = {
                "session_idx": int(idx),
                "animal_id": aid,
                "genotype": geno,
                "drug": drug,
                "cohort": f"{geno}_{drug}",
                "nominal_injection_frame_80hz": int(ev_row["nominal_injection_frame"]),
                "detected_removal_frame_80hz": int(ev_row["detected_removal_frame"]),
                "detected_return_frame_80hz": int(ev_row["detected_return_frame"]),
                "detected_entry_frame_80hz": int(ev_row["detected_entry_frame"]),
                "detected_return_frame_40hz": det_ret_40,
                "handling_duration_sec": float(ev_row["handling_duration_sec"]),
                "num_frames_40hz": len(syll_seq),
                "duration_minutes": len(syll_seq) / 40.0 / 60.0,
                "time_course_reference": "time_since_return",
                "original_session_id": matched_key,
            }

            lengths.append(len(syll_seq))
            for s in np.unique(syll_seq):
                unique_sylls.add(int(s))

    print("\nExtraction Summary (8KP Run 03 - Post-Return Alignment):")
    print(f"  - Total sessions extracted: {len(syllables_dict)}")
    print(f"  - Unique syllables observed: {len(unique_sylls)} (range {min(unique_sylls)}..{max(unique_sylls)})")
    print(f"  - Post-injection frames (40Hz): min={min(lengths)}, max={max(lengths)}, median={np.median(lengths):.1f}")
    print(f"  - Duration (min): min={min(lengths)/40/60:.1f}m, max={max(lengths)/40/60:.1f}m, median={np.median(lengths)/40/60:.1f}m")

    bundle = {
        "syllables_dict": syllables_dict,
        "pre_syllables_dict": pre_syllables_dict,
        "centroids_dict": centroids_dict,
        "metadata_dict": metadata_dict,
        "n_syllables": max(unique_sylls) + 1,
        "fps": 40.0,
    }

    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(bundle, f)

    print(f"\nSuccessfully saved 8KP shMoSeq input bundle to:\n  {OUTPUT_FILE}")
    print("=" * 80)


if __name__ == "__main__":
    main()
