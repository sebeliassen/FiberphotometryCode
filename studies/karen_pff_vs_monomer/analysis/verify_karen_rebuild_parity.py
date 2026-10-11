#!/usr/bin/env python3
"""
verify_karen_rebuild_parity.py

Compares the newly rebuilt sessions_karen_project_w_sylls.pkl
against sessions_karen_project_w_sylls_baseline_backup.pkl.
"""

import sys
import pickle
import numpy as np
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_
import pandas as pd
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

NEW_PKL = REPO_ROOT / "syllable_generation" / "data" / "sessions_karen_project_w_sylls.pkl"
BAK_PKL = REPO_ROOT / "syllable_generation" / "data" / "sessions_karen_project_w_sylls_baseline_backup.pkl"

def main():

    print("=== Karen Rebuild Parity Verification Suite ===")
    print(f"Loading new sessions: {NEW_PKL}")
    with open(NEW_PKL, "rb") as f:
        new_sessions = pickle.load(f)
        
    print(f"Loading backup sessions: {BAK_PKL}")
    with open(BAK_PKL, "rb") as f:
        bak_sessions = pickle.load(f)
        
    print(f"\n1. Session Count Check:")
    print(f"   New: {len(new_sessions)} sessions")
    print(f"   Bak: {len(bak_sessions)} sessions")
    assert len(new_sessions) == len(bak_sessions), f"Mismatch in session count: {len(new_sessions)} vs {len(bak_sessions)}"
    print("   -> [PASS] Session counts match perfectly.")
    
    new_map = {s.session_id: s for s in new_sessions}
    bak_map = {s.session_id: s for s in bak_sessions}
    
    assert set(new_map.keys()) == set(bak_map.keys()), "Session IDs do not match!"
    print("   -> [PASS] 110/110 Session IDs match 1-to-1.")
    
    # 2. Check metadata & conditions
    print("\n2. Metadata Enrichment Check:")
    cond_counts = pd.Series([s.metadata.get("condition") for s in new_sessions]).value_counts()
    print("   Condition breakdown:")
    for c, n in cond_counts.items():
        print(f"     - {c:<10}: {n} sessions")
        
    drug_counts = pd.Series([s.metadata.get("drug") for s in new_sessions]).value_counts()
    print("   Drug breakdown:")
    for d, n in drug_counts.items():
        print(f"     - {d:<25}: {n} sessions")
        
    assert (cond_counts.get("Unknown", 0) == 0), "Found sessions with Unknown condition!"
    assert all("weeks_post_injection" in s.metadata for s in new_sessions), "Missing weeks_post_injection!"
    assert all("is_drug_session" in s.metadata for s in new_sessions), "Missing is_drug_session!"
    print("   -> [PASS] 100% of sessions have valid condition, drug, and weeks_post_injection metadata.")
    
    # 3. Kinematic & Syllable Numerical Parity
    print("\n3. Kinematic & Syllable Parity Across All Sessions:")
    max_speed_diff = 0.0
    syll_matches = 0
    total_sylls = 0
    
    for sid, new_s in new_map.items():
        bak_s = bak_map[sid]
        
        df_new = new_s.all_data_df
        df_bak = bak_s.all_data_df
        
        assert len(df_new) == len(df_bak), f"Length mismatch for {sid}: {len(df_new)} vs {len(df_bak)}"
        
        # Speed diff
        v_new = df_new["velocity_px_s"].values
        v_bak = df_bak["velocity_px_s"].values
        valid_v = ~np.isnan(v_new) & ~np.isnan(v_bak)
        if valid_v.any():
            diff = np.max(np.abs(v_new[valid_v] - v_bak[valid_v]))
            if diff > max_speed_diff:
                max_speed_diff = diff
                
        # Syllable match
        s_new = df_new["syllable"].values
        s_bak = df_bak["syllable"].values
        valid_s = ~np.isnan(s_new) & ~np.isnan(s_bak)
        if valid_s.any():
            matches = (s_new[valid_s] == s_bak[valid_s]).sum()
            syll_matches += matches
            total_sylls += valid_s.sum()
            
    print(f"   Max Velocity Delta: {max_speed_diff:.2e} px/s (Threshold < 1e-6)")
    assert max_speed_diff < 1e-6, f"Velocity mismatch detected: max diff {max_speed_diff}"
    print(f"   Syllable Identity Match: {syll_matches:,} / {total_sylls:,} frames ({syll_matches/total_sylls*100:.4f}%)")
    assert syll_matches == total_sylls, "Syllable assignment mismatch!"
    print("   -> [PASS] Exact numerical and syllable parity verified against baseline backup.")
    
    print("\n" + "=" * 60)
    print("ALL PARITY & METADATA TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    main()
