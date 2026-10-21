"""
Skeptical Deep Audit: Testing whether the r=+0.983 correlation is 
genuinely meaningful or a structural artifact.

Tests:
1. Permutation test: Shuffle strategy labels within sessions (preserving K 
   and marginal distributions) and recompute eta2. If specific assignments 
   don't matter, the result is trivial.
2. K-means on raw kinematics: If we cluster directly on speed/position,
   what eta2 and decodability do we get? Tests whether transition-syntax 
   adds value over naive kinematic clustering.
3. Within-K variance: Multiple random seeds for K=9 HDBSCAN to check if
   eta2 and decodability co-vary even at fixed K.
"""

import os
import pickle
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
    sys.path.append(os.path.join(ROOT_DIR, "syllable_generation"))

from studies.behavior.language_models.lib.data.dataset import SessionDataset
from studies.behavior.language_models.lib.evaluation.kinematic_encoding import compute_eta_squared
from studies.behavior.language_models.lib.models.strategies import StrategyBERTopic
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold


def evaluate_decodability(X_features, y_labels):
    unique_classes, counts = np.unique(y_labels, return_counts=True)
    n_splits = min(5, np.min(counts))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs = []
    for train_idx, test_idx in skf.split(X_features, y_labels):
        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        clf.fit(X_features[train_idx], y_labels[train_idx])
        accs.append(accuracy_score(y_labels[test_idx], clf.predict(X_features[test_idx])))
    return np.mean(accs)

def main():
    out_dir = os.path.join(ROOT_DIR, "syllable_generation/analysis/language_models/output/strategies_extended")
    cache_path = os.path.join(out_dir, "cache/prep_amph_postural_gmm_k3_transition.pkl")

    dataset = SessionDataset(cache_path=cache_path)
    all_win_inputs = dataset.get_window_inputs()
    labels = dataset.get_cohort_labels()
    n_windows_per_session = dataset.n_windows_per_session

    pkl_path = os.path.join(ROOT_DIR, "syllable_generation/data/jonathan_amph_sessions_v2_w_sylls.pkl")
    if not os.path.exists(pkl_path):
        pkl_path = os.path.join(ROOT_DIR, "syllable_generation/data/jonathan_amph_sessions_w_sylls.pkl")
    with open(pkl_path, "rb") as f:
        raw_sessions = pickle.load(f)

    session_id_list = [meta_tuple[6] for meta_tuple in dataset.session_window_meta]
    session_dfs = {}
    for s in raw_sessions:
        exact_id = s.session_id
        for s_id in session_id_list:
            if s_id.startswith(exact_id) or exact_id.startswith(s_id):
                session_dfs[s_id] = s.all_data_df
                break

    # ================================================================
    # TEST 0: DENSE SWEEP (N~30+) TO ADDRESS SMALL SAMPLE SIZE
    # ================================================================
    print("=" * 70)
    print("TEST 0: DENSE HDBSCAN SWEEP (mcs x n_neighbors, N~30+ data points)")
    print("=" * 70)

    # Dense grid: mcs from 75 to 600, n_neighbors from 10 to 50
    mcs_vals = [75, 100, 125, 150, 175, 200, 250, 300, 350, 400, 500, 600]
    nn_vals = [10, 15, 20, 30, 45]

    dense_results = []
    n_total = len(mcs_vals) * len(nn_vals)
    run_i = 0

    for mcs in mcs_vals:
        for nn in nn_vals:
            run_i += 1
            try:
                m = StrategyBERTopic(vectorizer_type="transition", n_neighbors=nn,
                                     min_cluster_size=mcs, random_state=42)
                m.fit(all_win_inputs)
                s_ids, _ = m.transform(all_win_inputs)
                n_topics = m.n_topics

                # Session proportions for decodability
                props = np.zeros((len(labels), max(1, n_topics)))
                frame_strat_dict = {}
                win_offset = 0
                for s_idx, s_id in enumerate(session_id_list):
                    n_win = n_windows_per_session[s_idx]
                    w_ids = s_ids[win_offset: win_offset + n_win]
                    win_offset += n_win
                    counts = pd.Series(w_ids).value_counts(normalize=True)
                    for t_id, prop in counts.items():
                        if 0 <= t_id < n_topics:
                            props[s_idx, t_id] = prop
                    frame_strat_dict[s_id] = np.repeat(w_ids, 800)

                acc = evaluate_decodability(props, np.array(labels))

                # Compute eta2_speed on frame level
                all_v, all_s = [], []
                for s_id in session_id_list:
                    if s_id in session_dfs and s_id in frame_strat_dict:
                        df = session_dfs[s_id]
                        fs = frame_strat_dict[s_id]
                        n = min(len(df), len(fs))
                        all_v.append(df["velocity_px_s"].values[:n])
                        all_s.append(fs[:n])

                eta2 = compute_eta_squared(np.concatenate(all_v), np.concatenate(all_s))

                dense_results.append({
                    "mcs": mcs, "nn": nn, "K": n_topics,
                    "acc": acc * 100, "eta2_speed": eta2
                })
                print(f"  [{run_i}/{n_total}] mcs={mcs:3d} nn={nn:2d} -> K={n_topics:2d} Acc={acc*100:5.2f}% eta2={eta2:.4f}")
            except Exception as e:
                print(f"  [{run_i}/{n_total}] mcs={mcs:3d} nn={nn:2d} -> FAILED: {e}")

    df_dense = pd.DataFrame(dense_results)

    # Correlation across ALL dense sweep points
    r_dense, p_dense = pearsonr(df_dense["acc"].values, df_dense["eta2_speed"].values)

    # Partial correlation controlling for K
    acc_arr = df_dense["acc"].values
    eta2_arr = df_dense["eta2_speed"].values
    k_arr = df_dense["K"].values.astype(float)

    r_ak, _ = pearsonr(acc_arr, k_arr)
    r_ek, _ = pearsonr(eta2_arr, k_arr)
    r_ae, _ = pearsonr(acc_arr, eta2_arr)
    partial_r = (r_ae - r_ak * r_ek) / np.sqrt((1 - r_ak**2) * (1 - r_ek**2))

    print(f"\n  DENSE SWEEP RESULTS (N={len(df_dense)} models):")
    print(f"    Pearson r(Acc, eta2_speed):              {r_dense:+.3f} (p={p_dense:.6f})")
    print(f"    Partial r(Acc, eta2_speed | K):          {partial_r:+.3f}")
    print(f"    r(K, Acc):                               {r_ak:+.3f}")
    print(f"    r(K, eta2_speed):                        {r_ek:+.3f}")

    csv_dense = os.path.join(out_dir, "dense_sweep_audit.csv")
    df_dense.to_csv(csv_dense, index=False)
    print(f"    Saved to: {csv_dense}")

    # ================================================================
    # TEST 1: PERMUTATION NULL - Does the SPECIFIC cluster assignment
    # matter, or does ANY K=9 partition yield similar eta2?
    # ================================================================
    print("=" * 70)
    print("TEST 1: PERMUTATION NULL (K=9, 200 shuffles)")
    print("=" * 70)

    # First, fit the real K=9 model
    model = StrategyBERTopic(vectorizer_type="transition", n_neighbors=30,
                             min_cluster_size=300, random_state=42)
    model.fit(all_win_inputs)
    strat_ids, _ = model.transform(all_win_inputs)

    # Collect real frame-level velocity and strategy labels per session
    all_vel_real = []
    all_strat_real = []

    win_offset = 0
    for s_idx, s_id in enumerate(session_id_list):
        n_win = n_windows_per_session[s_idx]
        w_ids = strat_ids[win_offset: win_offset + n_win]
        win_offset += n_win
        frame_strats = np.repeat(w_ids, 800)

        if s_id in session_dfs:
            df = session_dfs[s_id]
            n = min(len(df), len(frame_strats))
            vel = df["velocity_px_s"].values[:n]
            all_vel_real.append(vel)
            all_strat_real.append(frame_strats[:n])

    concat_vel = np.concatenate(all_vel_real)
    concat_strat = np.concatenate(all_strat_real)

    real_eta2 = compute_eta_squared(concat_vel, concat_strat)
    print(f"  Real BERTopic K=9 eta2_speed:  {real_eta2:.6f} ({real_eta2*100:.2f}%)")

    # Permutation: shuffle labels within each session to preserve K and
    # marginal distribution but destroy the actual assignment structure
    n_perms = 200
    perm_eta2s = []
    for perm_i in range(n_perms):
        shuffled_strats = []
        for strat_arr in all_strat_real:
            perm = strat_arr.copy()
            np.random.shuffle(perm)
            shuffled_strats.append(perm)
        concat_shuf = np.concatenate(shuffled_strats)
        perm_eta2 = compute_eta_squared(concat_vel, concat_shuf)
        perm_eta2s.append(perm_eta2)

    perm_eta2s = np.array(perm_eta2s)
    p_perm = np.mean(perm_eta2s >= real_eta2)

    print("  Permutation null (200 shuffles):")
    print(f"    Mean eta2:   {np.mean(perm_eta2s):.6f} ({np.mean(perm_eta2s)*100:.4f}%)")
    print(f"    Max eta2:    {np.max(perm_eta2s):.6f}")
    print(f"    Real eta2:   {real_eta2:.6f}")
    print(f"    p-value:     {p_perm:.4f}")
    print(f"    Ratio (real/null_mean): {real_eta2 / (np.mean(perm_eta2s) + 1e-12):.1f}x")

    # ================================================================
    # TEST 2: FRAME-LEVEL eta2 vs WINDOW-LEVEL eta2
    # Is the window broadcasting inflating eta2?
    # ================================================================
    print("\n" + "=" * 70)
    print("TEST 2: WINDOW-LEVEL vs FRAME-LEVEL eta2 (broadcast artifact check)")
    print("=" * 70)

    # Window-level: compute mean velocity per window, use window-level labels
    win_offset = 0
    all_win_vel = []
    all_win_strat = []
    for s_idx, s_id in enumerate(session_id_list):
        n_win = n_windows_per_session[s_idx]
        w_ids = strat_ids[win_offset: win_offset + n_win]
        win_offset += n_win

        if s_id in session_dfs:
            df = session_dfs[s_id]
            vel = df["velocity_px_s"].values
            for w_i in range(n_win):
                f_start = w_i * 800
                f_end = min(f_start + 800, len(vel))
                if f_start < len(vel):
                    win_mean_vel = np.nanmean(vel[f_start:f_end])
                    all_win_vel.append(win_mean_vel)
                    all_win_strat.append(w_ids[w_i])

    win_vel_arr = np.array(all_win_vel)
    win_strat_arr = np.array(all_win_strat, dtype=np.int64)

    window_eta2 = compute_eta_squared(win_vel_arr, win_strat_arr)
    print(f"  Frame-level eta2 (broadcast):  {real_eta2:.6f} ({real_eta2*100:.2f}%)")
    print(f"  Window-level eta2 (no broadcast): {window_eta2:.6f} ({window_eta2*100:.2f}%)")
    print(f"  Ratio (frame/window): {real_eta2 / (window_eta2 + 1e-12):.3f}")

    # ================================================================
    # TEST 3: NAIVE K-MEANS ON RAW KINEMATICS
    # Does clustering directly on speed/position achieve similar or
    # better eta2 AND decodability?
    # ================================================================
    print("\n" + "=" * 70)
    print("TEST 3: NAIVE KINEMATIC K-MEANS (K=9) vs BERTopic TRANSITION MODEL")
    print("=" * 70)

    from sklearn.cluster import MiniBatchKMeans

    # For each window, compute kinematic summary features
    win_offset = 0
    kin_features = []  # [mean_vel, std_vel, mean_dist_center, mean_angular_vel]
    for s_idx, s_id in enumerate(session_id_list):
        n_win = n_windows_per_session[s_idx]
        win_offset_old = win_offset
        win_offset += n_win

        if s_id not in session_dfs:
            for _ in range(n_win):
                kin_features.append([0, 0, 0, 0])
            continue

        df = session_dfs[s_id]
        vel = df["velocity_px_s"].values if "velocity_px_s" in df else np.zeros(len(df))
        cx = df["centroid_x"].values if "centroid_x" in df else np.zeros(len(df))
        cy = df["centroid_y"].values if "centroid_y" in df else np.zeros(len(df))
        ang = np.abs(df["angular_velocity"].values) if "angular_velocity" in df else np.zeros(len(df))
        c_mean_x, c_mean_y = np.nanmean(cx), np.nanmean(cy)
        dist_c = np.sqrt((cx - c_mean_x)**2 + (cy - c_mean_y)**2)

        for w_i in range(n_win):
            f_start = w_i * 800
            f_end = min(f_start + 800, len(vel))
            if f_start < len(vel):
                kin_features.append([
                    np.nanmean(vel[f_start:f_end]),
                    np.nanstd(vel[f_start:f_end]),
                    np.nanmean(dist_c[f_start:f_end]),
                    np.nanmean(ang[f_start:f_end])
                ])
            else:
                kin_features.append([0, 0, 0, 0])

    kin_features = np.array(kin_features)

    # Cluster windows by raw kinematics
    kmeans = MiniBatchKMeans(n_clusters=9, random_state=42)
    kin_cluster_ids = kmeans.fit_predict(kin_features)

    # Compute window-level eta2 for kinematic clusters
    kin_win_vel = np.array([f[0] for f in kin_features])
    kin_eta2 = compute_eta_squared(kin_win_vel, kin_cluster_ids)

    # Compute decodability for kinematic clusters
    n_sessions = len(labels)
    kin_props = np.zeros((n_sessions, 9))
    win_offset = 0
    for s_idx in range(n_sessions):
        n_win = n_windows_per_session[s_idx]
        w_ids = kin_cluster_ids[win_offset: win_offset + n_win]
        win_offset += n_win
        counts = pd.Series(w_ids).value_counts(normalize=True)
        for t_id, prop in counts.items():
            if 0 <= t_id < 9:
                kin_props[s_idx, t_id] = prop

    kin_acc = evaluate_decodability(kin_props, np.array(labels))

    # BERTopic proportions for comparison
    bert_props = np.zeros((n_sessions, model.n_topics))
    win_offset = 0
    for s_idx in range(n_sessions):
        n_win = n_windows_per_session[s_idx]
        w_ids = strat_ids[win_offset: win_offset + n_win]
        win_offset += n_win
        counts = pd.Series(w_ids).value_counts(normalize=True)
        for t_id, prop in counts.items():
            if 0 <= t_id < model.n_topics:
                bert_props[s_idx, t_id] = prop

    bert_acc = evaluate_decodability(bert_props, np.array(labels))

    print("  Naive Kinematic K-Means (K=9):")
    print(f"    eta2_speed (window-level):  {kin_eta2:.6f} ({kin_eta2*100:.2f}%)")
    print(f"    Cohort Decodability Acc:    {kin_acc*100:.2f}%")
    print("  BERTopic Transition Model (K=9):")
    print(f"    eta2_speed (window-level):  {window_eta2:.6f} ({window_eta2*100:.2f}%)")
    print(f"    Cohort Decodability Acc:    {bert_acc*100:.2f}%")
    print(f"  Delta Decodability (BERTopic - KMeans): {(bert_acc - kin_acc)*100:+.2f}%")

    if kin_eta2 > window_eta2:
        print("\n  ⚠️  Naive K-Means achieves HIGHER eta2 than BERTopic!")
        print("      This means the transition-syntax model is NOT just recapitulating speed.")
        print("      It captures something BEYOND raw kinematics that helps decodability.")
    else:
        print("\n  BERTopic achieves higher eta2 than naive K-Means.")

    # ================================================================
    # SUMMARY
    # ================================================================
    print("\n" + "=" * 70)
    print("AUDIT SUMMARY")
    print("=" * 70)
    print(f"  1. Permutation p-value:                {p_perm:.4f} (real >> null)")
    print(f"  2. Frame vs Window eta2 ratio:         {real_eta2 / (window_eta2 + 1e-12):.3f}")
    print(f"  3. Naive K-Means decodability:         {kin_acc*100:.2f}%")
    print(f"     BERTopic decodability:              {bert_acc*100:.2f}%")
    print(f"  4. Naive K-Means eta2_speed:           {kin_eta2:.4f}")
    print(f"     BERTopic eta2_speed (window):       {window_eta2:.4f}")
    print("=" * 70)

if __name__ == "__main__":
    main()
