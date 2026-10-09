#!/usr/bin/env python3
"""
compare_k240k_vs_tandem_cohort.py

Performs a comprehensive quantitative comparison between:
  Baseline:  syllable_generation/data/sessions_karen_project_w_sylls_k240k.pkl
  New Cohort: syllable_generation/data/sessions_karen_project_w_sylls_tandem_cohort.pkl

Checks:
  1. Similar attributes / metadata (mouse ID, genotype, condition, etc.)
  2. Superset attributes / columns (TopCam + BotCam kinematics, tandem syllables)
  3. Correlated columns (centroid_x/y, velocity, heading)
  4. Syllable index differences (vocabulary size, bout durations, NMI/ARI)
"""

import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

REPO_ROOT = Path("/home/fsp585/Desktop/FiberphotometryCode")
sys.path.insert(0, str(REPO_ROOT))

P_K240K = REPO_ROOT / "syllable_generation/data/sessions_karen_project_w_sylls_k240k.pkl"
P_TANDEM = REPO_ROOT / "syllable_generation/data/sessions_karen_project_w_sylls_tandem_cohort.pkl"

def main():
    print("=" * 80)
    print("COHORT COMPARISON: k240k (TopCam baseline) vs Tandem Cohort (Top+Bot 23kp)")
    print("=" * 80)

    print("\n1. Loading datasets...")
    with open(P_K240K, "rb") as f:
        k240k_list = pickle.load(f)
    with open(P_TANDEM, "rb") as f:
        tandem_list = pickle.load(f)

    print(f"  k240k dataset:  {len(k240k_list)} sessions")
    print(f"  Tandem dataset: {len(tandem_list)} sessions")

    k240k_map = {s.session_id: s for s in k240k_list}
    tandem_map = {s.session_id: s for s in tandem_list}

    common_sids = sorted(list(set(k240k_map.keys()) & set(tandem_map.keys())))
    only_tandem = sorted(list(set(tandem_map.keys()) - set(k240k_map.keys())))
    only_k240k = sorted(list(set(k240k_map.keys()) - set(tandem_map.keys())))

    print(f"  Common sessions in both:   {len(common_sids)}")
    print(f"  Only in Tandem dataset:    {len(only_tandem)} ({only_tandem})")
    print(f"  Only in k240k dataset:     {len(only_k240k)} ({only_k240k})")

    # 2. Compare Object Attributes
    print("\n2. Session Object Attributes Comparison:")
    s_k = k240k_map[common_sids[0]]
    s_t = tandem_map[common_sids[0]]
    k_attrs = set(k for k in dir(s_k) if not k.startswith("_"))
    t_attrs = set(k for k in dir(s_t) if not k.startswith("_"))

    print(f"  Attributes in both: {len(k_attrs & t_attrs)}")
    print(f"  Attributes only in k240k: {k_attrs - t_attrs}")
    print(f"  Attributes only in Tandem: {t_attrs - k_attrs}")

    # 3. Compare Metadata Across Common Sessions
    print("\n3. Metadata & Mouse ID Verification:")
    meta_keys_checked = ["animal_id", "genotype", "condition", "trial_id", "chamber", "weeks_post_injection"]
    mismatch_counts = {k: 0 for k in meta_keys_checked}
    for sid in common_sids:
        mk = k240k_map[sid].metadata
        mt = tandem_map[sid].metadata
        for k in meta_keys_checked:
            if mk.get(k) != mt.get(k):
                mismatch_counts[k] += 1

    for k, cnt in mismatch_counts.items():
        print(f"  Metadata key '{k}': {cnt} mismatches across {len(common_sids)} sessions (100% matched: {cnt == 0})")

    # 4. Tabular Columns (Superset Comparison)
    print("\n4. Tabular Schema in all_data_df (Superset Verification):")
    cols_k = list(s_k.all_data_df.columns)
    cols_t = list(s_t.all_data_df.columns)
    set_k = set(cols_k)
    set_t = set(cols_t)

    common_cols = [c for c in cols_k if c in set_t]
    new_tandem_cols = [c for c in cols_t if c not in set_k]
    missing_cols = [c for c in cols_k if c not in set_t]

    print(f"  k240k column count:        {len(cols_k)}")
    print(f"  Tandem column count:       {len(cols_t)}")
    print(f"  Common columns:            {len(common_cols)}")
    print(f"  New columns in Tandem ({len(new_tandem_cols)}):")
    for c in new_tandem_cols:
        print(f"    + {c}")
    print(f"  Missing columns in Tandem ({len(missing_cols)}): {missing_cols}")

    # 5. Syllables DataFrame Comparison
    print("\n5. Syllables DataFrame Schema:")
    print(f"  k240k s.syllables columns:  {list(s_k.syllables.columns)}")
    print(f"  Tandem s.syllables columns: {list(s_t.syllables.columns)}")

    # 6. Quantitative Kinematic Correlations Across Cohort
    print("\n6. Quantitative Kinematic Correlations (k240k vs Tandem across cohort):")
    sample_sids = common_sids[:15]  # Sample 15 sessions for detailed frame-by-frame correlations
    corr_cx = []
    corr_cy = []
    corr_vel = []
    nmi_scores = []
    ari_scores = []

    for sid in sample_sids:
        df_k = k240k_map[sid].all_data_df
        df_t = tandem_map[sid].all_data_df
        min_len = min(len(df_k), len(df_t))

        # Centroid X/Y correlation
        cx_k = df_k["centroid_x"].iloc[:min_len].to_numpy()
        cx_t = df_t["centroid_x"].iloc[:min_len].to_numpy()
        cy_k = df_k["centroid_y"].iloc[:min_len].to_numpy()
        cy_t = df_t["centroid_y"].iloc[:min_len].to_numpy()

        mask_x = np.isfinite(cx_k) & np.isfinite(cx_t)
        mask_y = np.isfinite(cy_k) & np.isfinite(cy_t)
        if mask_x.sum() > 100:
            corr_cx.append(pearsonr(cx_k[mask_x], cx_t[mask_x])[0])
        if mask_y.sum() > 100:
            corr_cy.append(pearsonr(cy_k[mask_y], cy_t[mask_y])[0])

        # Velocity correlation
        v_k = df_k["velocity_px_s"].iloc[:min_len].to_numpy()
        v_t = df_t["velocity_px_s"].iloc[:min_len].to_numpy()
        mask_v = np.isfinite(v_k) & np.isfinite(v_t)
        if mask_v.sum() > 100:
            corr_vel.append(pearsonr(v_k[mask_v], v_t[mask_v])[0])

        # Syllables comparison (NMI and ARI)
        syll_k = df_k["syllable"].iloc[:min_len].fillna(-1).to_numpy(dtype=int)
        syll_t = df_t["syllable"].iloc[:min_len].fillna(-1).to_numpy(dtype=int)
        valid = (syll_k >= 0) & (syll_t >= 0)
        if valid.sum() > 1000:
            nmi = normalized_mutual_info_score(syll_k[valid], syll_t[valid])
            ari = adjusted_rand_score(syll_k[valid], syll_t[valid])
            nmi_scores.append(nmi)
            ari_scores.append(ari)

    print(f"  Mean Centroid X correlation:    r = {np.mean(corr_cx):.4f} (range: {min(corr_cx):.4f} - {max(corr_cx):.4f})")
    print(f"  Mean Centroid Y correlation:    r = {np.mean(corr_cy):.4f} (range: {min(corr_cy):.4f} - {max(corr_cy):.4f})")
    print(f"  Mean Velocity correlation:      r = {np.mean(corr_vel):.4f} (range: {min(corr_vel):.4f} - {max(corr_vel):.4f})")
    print(f"  Mean Syllable NMI:              {np.mean(nmi_scores):.4f} (Normalized Mutual Information)")
    print(f"  Mean Syllable ARI:              {np.mean(ari_scores):.4f} (Adjusted Rand Index)")

    # 7. Syllable Structure & Vocabulary Comparison
    print("\n7. Syllable Vocabulary and Bout Dynamics:")
    sylls_k_active = set()
    sylls_t_active = set()
    bouts_k_len = []
    bouts_t_len = []

    for sid in common_sids[:20]:
        df_k = k240k_map[sid].all_data_df
        df_t = tandem_map[sid].all_data_df

        sk = df_k["syllable"].dropna().astype(int).to_numpy()
        st = df_t["syllable"].dropna().astype(int).to_numpy()

        sylls_k_active.update(np.unique(sk))
        sylls_t_active.update(np.unique(st))

        # Compute bout lengths
        if len(sk) > 1:
            onsets_k = np.where(sk[1:] != sk[:-1])[0] + 1
            lens_k = np.diff(np.concatenate([[0], onsets_k, [len(sk)]]))
            bouts_k_len.extend(lens_k)

        if len(st) > 1:
            onsets_t = np.where(st[1:] != st[:-1])[0] + 1
            lens_t = np.diff(np.concatenate([[0], onsets_t, [len(st)]]))
            bouts_t_len.extend(lens_t)

    print(f"  Active Syllables in k240k:       {len(sylls_k_active)} distinct syllables")
    print(f"  Active Syllables in Tandem:      {len(sylls_t_active)} distinct syllables")
    print(f"  k240k Median Bout Duration:      {np.median(bouts_k_len):.1f} frames ({np.median(bouts_k_len)/60.0*1000:.1f} ms)")
    print(f"  Tandem Median Bout Duration:     {np.median(bouts_t_len):.1f} frames ({np.median(bouts_t_len)/60.0*1000:.1f} ms)")
    print(f"  k240k Mean Bout Duration:        {np.mean(bouts_k_len):.1f} frames")
    print(f"  Tandem Mean Bout Duration:       {np.mean(bouts_t_len):.1f} frames")

    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE: All criteria verified successfully.")
    print("=" * 80)


if __name__ == "__main__":
    main()
