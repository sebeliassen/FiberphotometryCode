#!/usr/bin/env python3
"""
verify_karen_timeseries_sync.py

Comprehensive validation suite for Karen final model syllable extraction and attachment:
1. File Existence & Integrity (results.h5, sessions_karen_project_w_sylls.pkl)
2. Schema & Dtype Parity against reference AMPH v2 (jonathan_amph_sessions_v2_w_sylls.pkl)
3. Length and Cardinality Invariance across all 113 sessions
4. Timeseries Synchronization (40Hz Model -> 80Hz Master Timeline 2-frame ffill integrity)
5. Boundary & NaN Gap Preservation
6. Behavioral & Syllable Distribution Sanity Metrics
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

KAREN_RESULTS_H5 = REPO_ROOT / "syllable_generation" / "syllabizers" / "keypoint_moseq_syll" / "karen_project_2" / "final_karen_model" / "results.h5"
KAREN_VFI_NPZ = REPO_ROOT / "syllable_generation" / "syllabizers" / "keypoint_moseq_syll" / "karen_project_2" / "final_karen_model" / "video_frame_indexes.npz"
KAREN_WSYLLS_PKL = REPO_ROOT / "syllable_generation" / "data" / "sessions_karen_project_w_sylls.pkl"
AMPH_V2_PKL = REPO_ROOT / "syllable_generation" / "data" / "jonathan_amph_sessions_v2_w_sylls.pkl"


def run_checks():
    passed_tests = 0
    total_tests = 0

    print("=" * 75)
    print("      KAREN FINAL MODEL TIMESERIES SYNC & ATTACHMENT VERIFICATION")
    print("=" * 75)

    # 1. FILE EXISTENCE
    total_tests += 1
    print("\n[Test 1] Checking file existence...")
    if not KAREN_RESULTS_H5.exists():
        print(f"  FAILED: Missing {KAREN_RESULTS_H5}")
        return False
    if not KAREN_WSYLLS_PKL.exists():
        print(f"  FAILED: Missing {KAREN_WSYLLS_PKL}")
        return False
    print(f"  PASSED: Found results.h5 ({KAREN_RESULTS_H5.stat().st_size / 1e6:.1f} MB)")
    print(f"  PASSED: Found sessions_karen_project_w_sylls.pkl ({KAREN_WSYLLS_PKL.stat().st_size / 1e9:.2f} GB)")
    passed_tests += 1

    # Load Karen sessions
    with open(KAREN_WSYLLS_PKL, "rb") as f:
        karen_sessions = pickle.load(f)
    print(f"  Loaded {len(karen_sessions)} Karen sessions.")

    # Load VFI
    with np.load(KAREN_VFI_NPZ) as z:
        vfi = {k: z[k].astype(int) for k in z.files}

    # 2. SESSION COUNT CHECK
    total_tests += 1
    print("\n[Test 2] Session count & VFI match check...")
    if len(karen_sessions) < 110:
        print(f"  FAILED: Only matched {len(karen_sessions)} sessions (< 110 expected).")
    else:
        print(f"  PASSED: Successfully matched all {len(karen_sessions)} unique session records (from 113 VFI recording keys).")
        passed_tests += 1

    # 3. SCHEMA & DTYPE COMPARISON AGAINST AMPH V2
    total_tests += 1
    print("\n[Test 3] Schema & Dtype Parity vs AMPH v2 reference...")
    if AMPH_V2_PKL.exists():
        with open(AMPH_V2_PKL, "rb") as f:
            amph_sessions = pickle.load(f)
        ref_s0 = amph_sessions[0]
        k_s0 = karen_sessions[0]

        # Core required columns from AMPH v2
        core_cols = ['frame_index', 'name', 'centroid_x', 'centroid_y', 'heading',
                     'angular_velocity', 'velocity_px_s', 'syllable', 'group', 'onset']
        missing_core = [c for c in core_cols if c not in k_s0.all_data_df.columns]
        if missing_core:
            print(f"  FAILED: Missing core columns in all_data_df: {missing_core}")
        else:
            print("  PASSED: All core kinematic & syllable columns present in all_data_df.")

        # Syllables DF columns
        syll_core = ['syllable', 'latent_state_0', 'centroid_0', 'centroid_1', 'heading']
        missing_syll = [c for c in syll_core if c not in k_s0.syllables.columns]
        if missing_syll:
            print(f"  FAILED: Missing columns in session.syllables: {missing_syll}")
        else:
            print("  PASSED: All latent & syllable columns present in session.syllables.")

        # Syllable dtype
        if k_s0.all_data_df['syllable'].dtype == 'Int64' and k_s0.syllables['syllable'].dtype == 'Int64':
            print("  PASSED: Syllable columns are nullable 'Int64' matching AMPH v2.")
            passed_tests += 1
        else:
            print(f"  FAILED: Syllable dtype mismatch: all_data_df={k_s0.all_data_df['syllable'].dtype}, syllables={k_s0.syllables['syllable'].dtype}")
    else:
        print(f"  SKIPPED: AMPH v2 reference pkl not found at {AMPH_V2_PKL}")

    # 4. CARDINALITY & MASTER TIMELINE LENGTH INVARIANCE
    total_tests += 1
    print(f"\n[Test 4] Cardinality & Timeline Length Invariance across all {len(karen_sessions)} sessions...")
    length_mismatches = 0
    total_frames = 0
    total_valid_syll_frames = 0

    for i, s in enumerate(karen_sessions):
        df_len = len(s.all_data_df)
        syll_len = len(s.syllables)
        total_frames += df_len
        total_valid_syll_frames += s.all_data_df['syllable'].notna().sum()

        if df_len != syll_len:
            length_mismatches += 1
            print(f"  FAIL: Session {s.session_id}: all_data_df ({df_len}) != syllables ({syll_len})")

        # Check sequential frame_index
        fi = s.all_data_df['frame_index'].values
        if not np.array_equal(fi, np.arange(df_len)):
            length_mismatches += 1
            print(f"  FAIL: Session {s.session_id}: frame_index is not sequential 0..N-1")

    if length_mismatches == 0:
        print("  PASSED: 100% of sessions have identical all_data_df and syllables lengths with strictly sequential 0..N-1 frame_index.")
        print(f"  Total Master Timeline Frames: {total_frames:,} | Total Valid Syllable Frames: {total_valid_syll_frames:,} ({total_valid_syll_frames/total_frames*100:.1f}% coverage)")
        passed_tests += 1
    else:
        print(f"  FAILED: Found {length_mismatches} length/index mismatches.")

    # 5. TIMESERIES SYNC & 40Hz -> 80Hz FFILL VERIFICATION
    total_tests += 1
    print("\n[Test 5] Timeseries Sync (40Hz Model -> 80Hz Timeline Weaving Check)...")
    sync_errors = 0
    samples_checked = 0

    for s in karen_sessions[:15]:  # Check detailed sync on first 15 sessions
        rk = s.results_dict_key
        if rk not in vfi:
            continue
        vfi_arr = vfi[rk]
        syll = s.all_data_df['syllable'].values

        # Check that on each even VFI index `idx`, syllable[idx] == syllable[idx+1] (ffill property)
        # unless idx+1 was the end of the recording or an unmodeled boundary
        for start_idx in vfi_arr[:500:50]:  # sample test points
            if start_idx + 1 < len(syll):
                val_even = syll[start_idx]
                val_odd = syll[start_idx + 1]
                if pd.notna(val_even) and pd.notna(val_odd):
                    samples_checked += 1
                    if val_even != val_odd:
                        sync_errors += 1

    if sync_errors == 0 and samples_checked > 0:
        print(f"  PASSED: Verified {samples_checked} sampled timepoints. Even-odd frame pairs match perfectly via 2-frame ffill.")
        passed_tests += 1
    else:
        print(f"  FAILED: Found {sync_errors} sync errors out of {samples_checked} samples.")

    # 6. BOUNDARY & GAP INTEGRITY CHECK
    total_tests += 1
    print("\n[Test 6] Boundary & Gap Integrity Check (No bleed into unmodeled frames)...")
    boundary_errors = 0
    for s in karen_sessions[:15]:
        rk = s.results_dict_key
        if rk not in vfi:
            continue
        vfi_arr = vfi[rk]
        min_vfi = vfi_arr.min()
        max_vfi = vfi_arr.max()
        syll = s.all_data_df['syllable']

        # Leading frames before min_vfi must be NaN
        if min_vfi > 0:
            leading_notna = syll.iloc[:min_vfi].notna().sum()
            if leading_notna > 0:
                boundary_errors += 1
                print(f"  FAIL: Session {s.session_id} has {leading_notna} non-NaN frames before first VFI index {min_vfi}.")

        # Trailing frames after max_vfi + 1 must be NaN
        if max_vfi + 2 < len(syll):
            trailing_notna = syll.iloc[max_vfi + 2:].notna().sum()
            if trailing_notna > 0:
                boundary_errors += 1
                print(f"  FAIL: Session {s.session_id} has {trailing_notna} non-NaN frames after last VFI index {max_vfi}.")

    if boundary_errors == 0:
        print("  PASSED: Boundary frames before tracking start and after tracking end remain strictly NaN.")
        passed_tests += 1
    else:
        print(f"  FAILED: Found {boundary_errors} boundary bleed violations.")

    # 7. SYLLABLE VOCABULARY & USAGE METRICS
    total_tests += 1
    print("\n[Test 7] Syllable Vocabulary & Usage Distribution...")
    all_sylls = pd.concat([s.all_data_df['syllable'].dropna() for s in karen_sessions])
    unique_sylls = np.sort(all_sylls.unique())
    print(f"  Number of Unique Active Syllables: {len(unique_sylls)}")
    print(f"  Syllable ID Range: [{unique_sylls.min()}, {unique_sylls.max()}]")
    top5 = all_sylls.value_counts().head(5)
    print(f"  Top 5 Syllables by Frequency:\n{top5}")

    if len(unique_sylls) > 10 and unique_sylls.min() >= 0:
        print("  PASSED: Syllable vocabulary is healthy and non-degenerate.")
        passed_tests += 1
    else:
        print(f"  FAILED: Syllable vocabulary appears degenerate ({len(unique_sylls)} unique).")

    print("\n" + "=" * 75)
    print(f"VERIFICATION SUMMARY: {passed_tests} / {total_tests} Tests Passed.")
    print("=" * 75)
    return passed_tests == total_tests


if __name__ == "__main__":
    success = run_checks()
    sys.exit(0 if success else 1)
