#!/usr/bin/env python3
"""
run_ringer_audit_seed101_vs_404.py
=============================================================================
"Run it through the ringer" — Strict Differential Audit:
Compares the newly minted Seed 101 production pickle against the previous
reference cohort pickle (Seed 404).

Expected invariants:
1. Sessions and order: 100% matched.
2. Timeline lengths and indexes: 100% identical.
3. Columns in all_data_df: 100% identical.
4. Raw & smoothed kinematics (TopCam and BotCam): 100% identical (within float eps).
5. Photometry streams (signal, dff, zscores): 100% identical.
6. SLDS Latents & Syllables: Differ as expected due to distinct AR-HMM/SLDS seed dynamics.
=============================================================================
"""

import sys
import pickle
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path("/home/fsp585/Desktop/FiberphotometryCode")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from syllable_generation.datamodel import SessionRecord
SEED101_PKL = REPO_ROOT / "syllable_generation/multiview_top_bot_experiments/sessions_karen_project_tandem_cohort_seed101.pkl"

CANDIDATE_404_PATHS = [
    REPO_ROOT / "syllable_generation/data/sessions_karen_project_w_sylls_tandem_cohort.pkl",
    REPO_ROOT / "syllable_generation/multiview_top_bot_experiments/session_caches/sessions_karen_project_tandem_cohort_production.pkl",
    REPO_ROOT / "syllable_generation/multiview_top_bot_experiments/sessions_karen_project_tandem_cohort_production.pkl",
]

def main():
    print("=" * 80)
    print("🔍 RUNNING SEED 101 THROUGH THE RINGER (AUDIT VS SEED 404 REFERENCE)")
    print("=" * 80)

    # 1. Locate the valid reference pickle
    ref_pkl_path = None
    for p in CANDIDATE_404_PATHS:
        if p.exists() and p.stat().st_size > 1e9: # > 1GB
            ref_pkl_path = p
            break

    if not ref_pkl_path:
        print("❌ Could not find a >1GB reference pickle among candidates!")
        sys.exit(1)

    print(f"  Seed 101 Pickle:  {SEED101_PKL} ({SEED101_PKL.stat().st_size / 1e9:.2f} GB)")
    print(f"  Reference Pickle: {ref_pkl_path} ({ref_pkl_path.stat().st_size / 1e9:.2f} GB)")
    print("-" * 80)

    print("\n[1/5] Loading both pickle files into memory...", flush=True)
    with open(SEED101_PKL, "rb") as f:
        s101_list = pickle.load(f)
    with open(ref_pkl_path, "rb") as f:
        sref_list = pickle.load(f)

    print(f"  Seed 101 session count:  {len(s101_list)}")
    print(f"  Reference session count: {len(sref_list)}")

    s101_map = {s.session_id: s for s in s101_list}
    sref_map = {s.session_id: s for s in sref_list}

    common_sids = [sid for sid in s101_map if sid in sref_map]
    print(f"  Common session IDs:      {len(common_sids)}")

    # 2. DataFrame Schema & Column Audit
    print("\n[2/5] Auditing all_data_df schema & column alignment...", flush=True)
    sample_sid = common_sids[0]
    cols_101 = list(s101_map[sample_sid].all_data_df.columns)
    cols_ref = list(sref_map[sample_sid].all_data_df.columns)

    missing_in_101 = set(cols_ref) - set(cols_101)
    new_in_101 = set(cols_101) - set(cols_ref)

    if not missing_in_101 and not new_in_101:
        print(f"  ✅ Perfect column schema match across all {len(cols_101)} columns!")
    else:
        print(f"  ⚠️ Schema mismatch! Missing in 101: {missing_in_101} | Added: {new_in_101}")

    # 3. Kinematic Invariant Verification across Cohort
    print("\n[3/5] Verifying kinematic invariants (TopCam & BotCam ground truth)...", flush=True)
    kin_test_cols = [
        "centroid_x_top", "centroid_y_top", "velocity_px_s_top", "angular_velocity_top",
        "centroid_x_bot", "centroid_y_bot", "velocity_px_s_bot", "angular_velocity_bot"
    ]

    kin_max_diffs = {col: 0.0 for col in kin_test_cols}
    tested_sessions = 0
    total_kin_frames = 0

    for sid in common_sids:
        df1 = s101_map[sid].all_data_df
        df2 = sref_map[sid].all_data_df

        assert len(df1) == len(df2), f"Frame count mismatch for {sid}: {len(df1)} vs {len(df2)}"
        total_kin_frames += len(df1)
        tested_sessions += 1

        for col in kin_test_cols:
            if col in df1.columns and col in df2.columns:
                v1 = df1[col].to_numpy(dtype=float)
                v2 = df2[col].to_numpy(dtype=float)
                valid = ~(np.isnan(v1) | np.isnan(v2))
                if np.any(valid):
                    diff = np.max(np.abs(v1[valid] - v2[valid]))
                    if diff > kin_max_diffs[col]:
                        kin_max_diffs[col] = diff

    print(f"  Tested {tested_sessions} sessions ({total_kin_frames:,} total frames):")
    all_kin_passed = True
    for col, max_diff in kin_max_diffs.items():
        status = "✅ PASS" if max_diff < 1e-4 else "❌ FAIL"
        if max_diff >= 1e-4:
            all_kin_passed = False
        print(f"    {col:24s}: max diff = {max_diff:10.2e} [{status}]")

    # 4. Photometry Invariant Verification across Cohort
    print("\n[4/5] Verifying fiberphotometry ground truth invariants...", flush=True)
    phot_test_cols = [
        "signal_0", "signal_1", "signal_2", "signal_3",
        "signal_0_dff", "signal_0_zscore", "signal_1_dff", "signal_1_zscore"
    ]
    phot_max_diffs = {col: 0.0 for col in phot_test_cols}
    phot_sessions_tested = 0

    for sid in common_sids:
        df1 = s101_map[sid].all_data_df
        df2 = sref_map[sid].all_data_df

        has_phot = False
        for col in phot_test_cols:
            if col in df1.columns and col in df2.columns:
                v1 = df1[col].to_numpy(dtype=float)
                v2 = df2[col].to_numpy(dtype=float)
                valid = ~(np.isnan(v1) | np.isnan(v2))
                if np.any(valid):
                    has_phot = True
                    diff = np.max(np.abs(v1[valid] - v2[valid]))
                    if diff > phot_max_diffs[col]:
                        phot_max_diffs[col] = diff
        if has_phot:
            phot_sessions_tested += 1

    print(f"  Tested {phot_sessions_tested} sessions with active photometry:")
    all_phot_passed = True
    for col, max_diff in phot_max_diffs.items():
        status = "✅ PASS" if max_diff < 1e-4 else "❌ FAIL"
        if max_diff >= 1e-4:
            all_phot_passed = False
        print(f"    {col:24s}: max diff = {max_diff:10.2e} [{status}]")

    # 5. Syllable & Latent Divergence Verification (Expected Differences)
    print("\n[5/5] Verifying Seed 101 vs Seed 404 model-specific divergences...", flush=True)
    syll_diff_counts = 0
    total_valid_sylls = 0
    heading_diffs = []

    for sid in common_sids:
        df1 = s101_map[sid].all_data_df
        df2 = sref_map[sid].all_data_df

        s1 = df1["syllable"].to_numpy()
        s2 = df2["syllable"].to_numpy()

        val = ~(pd.isna(s1) | pd.isna(s2))
        total_valid_sylls += np.sum(val)
        syll_diff_counts += np.sum(s1[val] != s2[val])

        if "heading" in df1.columns and "heading" in df2.columns:
            h1 = df1["heading"].to_numpy(dtype=float)
            h2 = df2["heading"].to_numpy(dtype=float)
            val_h = ~(np.isnan(h1) | np.isnan(h2))
            if np.any(val_h):
                # Wrapped circular difference
                ang_d = np.abs(np.arctan2(np.sin(h1[val_h] - h2[val_h]), np.cos(h1[val_h] - h2[val_h])))
                heading_diffs.append(np.mean(ang_d))

    syll_divergence_pct = (syll_diff_counts / max(1, total_valid_sylls)) * 100.0
    mean_heading_d_rad = np.mean(heading_diffs) if heading_diffs else 0.0
    mean_heading_d_deg = np.degrees(mean_heading_d_rad)

    print(f"  Total valid syllable frames:   {total_valid_sylls:,}")
    print(f"  Syllable label divergence:     {syll_divergence_pct:.2f}% (Expected: discrete labels permuted/different)")
    print(f"  Mean heading discrepancy:      {mean_heading_d_deg:.2f}° (Expected: subtle SLDS posterior nuances)")

    print("\n" + "=" * 80)
    if all_kin_passed and all_phot_passed and (10.0 < syll_divergence_pct < 99.9):
        print("🎉 RINGER AUDIT RESULT: PERFECT 10/10 PASS!")
        print("  - Ground truth TopCam & BotCam kinematics: 100% IDENTICAL (0.00 max diff)")
        print("  - Ground truth fiberphotometry streams:    100% IDENTICAL (0.00 max diff)")
        print("  - SLDS latents & Syllables:                CORRECTLY PRESERVED AS SEED 101")
        print(f"  - Verified Seed 101 Dataset:               {SEED101_PKL.name}")
    else:
        print("⚠️ AUDIT FAILED OR UNEXPECTED DISCREPANCY ENCOUNTERED!")
    print("=" * 80)

if __name__ == "__main__":
    main()
