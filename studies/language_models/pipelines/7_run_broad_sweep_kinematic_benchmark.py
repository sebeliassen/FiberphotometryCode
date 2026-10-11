import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

# Paths & Settings
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
    sys.path.append(os.path.join(ROOT_DIR, "syllable_generation"))

from studies.behavior.language_models.lib.data.dataset import SessionDataset
from studies.behavior.language_models.lib.evaluation.kinematic_encoding import evaluate_session_kinematic_encoding
from studies.behavior.language_models.lib.models.strategies import StrategyBERTopic
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold


def compute_entropy_and_keff(proportions_matrix: np.ndarray):
    mean_props = np.mean(proportions_matrix, axis=0)
    p = mean_props[mean_props > 0]
    p = p / np.sum(p)

    K_active = len(p)
    if K_active <= 1:
        return K_active, 0.0, 1.0

    H = -np.sum(p * np.log2(p))
    H_norm = H / np.log2(K_active)
    K_eff = 2 ** H
    return K_active, H_norm, K_eff

def evaluate_classification(X_features: np.ndarray, y_labels: np.ndarray):
    unique_classes, counts = np.unique(y_labels, return_counts=True)
    n_splits = min(5, np.min(counts))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    accs, f1s = [], []
    for train_idx, test_idx in skf.split(X_features, y_labels):
        X_train, X_test = X_features[train_idx], X_features[test_idx]
        y_train, y_test = y_labels[train_idx], y_labels[test_idx]

        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        accs.append(accuracy_score(y_test, y_pred))
        f1s.append(f1_score(y_test, y_pred, average='weighted', zero_division=0))

    return np.mean(accs), np.mean(f1s)

def main():
    log_dir = os.path.join(ROOT_DIR, "syllable_generation/analysis/language_models/output/strategies_extended/logs")
    out_dir = os.path.join(ROOT_DIR, "syllable_generation/analysis/language_models/output/strategies_extended")
    cache_path = os.path.join(out_dir, "cache/prep_amph_postural_gmm_k3_transition.pkl")
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, "broad_sweep_kinematic_benchmark.log")
    with open(log_file, "w") as f:
        f.write("=== 45-Run Broad Strategy Scale Sweep Benchmark ===\n")

    def log_msg(msg: str):
        print(msg)
        with open(log_file, "a") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")

    log_msg("=== Starting 45-Run Broad Strategy Scale Sweep (Kinematic Encoding & Correlation Benchmark) ===")

    dataset = SessionDataset(cache_path=cache_path)
    all_win_inputs = dataset.get_window_inputs()
    labels = dataset.get_cohort_labels()
    n_windows_per_session = dataset.n_windows_per_session

    # Load raw session dataframes
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

    results = []

    # ---------------------------------------------------------
    # APPROACH A: 2-Tier Hierarchical Merging (15 Runs)
    # ---------------------------------------------------------
    print("\n--- Running Approach A: 2-Tier Hierarchical Merging ---")
    base_model = StrategyBERTopic(vectorizer_type="transition", n_neighbors=30, min_cluster_size=75)
    base_model.fit(all_win_inputs)
    base_strat_ids, _ = base_model.transform(all_win_inputs)
    n_base_topics = base_model.n_topics
    X_vec = base_model._prepare_vectors(all_win_inputs)

    base_topic_vectors = []
    for t in range(n_base_topics):
        mask = (base_strat_ids == t)
        if np.any(mask):
            base_topic_vectors.append(X_vec[mask].mean(axis=0))
        else:
            base_topic_vectors.append(np.zeros(X_vec.shape[1]))
    base_topic_vectors = np.array(base_topic_vectors) + 1e-9

    for k_super in [4, 6, 8, 10, 12]:
        for metric in ["cosine", "euclidean", "average"]:
            run_name = f"ApproachA_Hierarchy_K{k_super}_{metric}"
            try:
                if metric == "average":
                    agg = AgglomerativeClustering(n_clusters=k_super, metric="cosine", linkage="average")
                elif metric == "cosine":
                    agg = AgglomerativeClustering(n_clusters=k_super, metric="cosine", linkage="complete")
                else:
                    agg = AgglomerativeClustering(n_clusters=k_super, metric="euclidean", linkage="ward")

                super_map = agg.fit_predict(base_topic_vectors)
                super_win_ids = np.array([super_map[t] if t < len(super_map) else 0 for t in base_strat_ids])

                session_props = []
                frame_strat_dict = {}
                win_offset = 0
                for s_idx, s_id in enumerate(session_id_list):
                    n_win = n_windows_per_session[s_idx]
                    if n_win == 0:
                        session_props.append(np.zeros(k_super))
                        continue
                    w_ids = super_win_ids[win_offset : win_offset + n_win]
                    win_offset += n_win
                    counts = pd.Series(w_ids).value_counts()
                    p = [counts.get(t, 0) / n_win for t in range(k_super)]
                    session_props.append(p)
                    frame_strat_dict[s_id] = np.repeat(w_ids, 800)

                X_props = np.array(session_props)
                acc, f1 = evaluate_classification(X_props, labels)
                k_act, h_norm, k_eff = compute_entropy_and_keff(X_props)
                kin_metrics = evaluate_session_kinematic_encoding(session_dfs, frame_strat_dict)

                results.append({
                    "approach": "Approach A (Hierarchy)",
                    "run_name": run_name,
                    "target_k": k_super,
                    "active_k": k_act,
                    "group_decodability_acc_pct": acc * 100.0,
                    "weighted_f1": f1,
                    "entropy_norm": h_norm,
                    "k_effective": k_eff,
                    "eta2_speed": kin_metrics["eta2_velocity"],
                    "eta2_dist_center": kin_metrics["eta2_dist_center"],
                    "eta2_angular_vel": kin_metrics["eta2_angular_vel"],
                    "eta2_cadence": kin_metrics["eta2_cadence"]
                })
                print(f"  [A] K={k_super:2d} | Metric={metric:9s} --> Acc: {acc*100:5.2f}% | eta2_speed: {kin_metrics['eta2_velocity']:.3f} | eta2_dist: {kin_metrics['eta2_dist_center']:.3f}")
            except Exception as exc:
                print(f"  [A] Error in {run_name}: {exc}")

    # ---------------------------------------------------------
    # APPROACH B: HDBSCAN Scale Tuning (21 Runs)
    # ---------------------------------------------------------
    print("\n--- Running Approach B: HDBSCAN Scale Tuning ---")
    for mcs in [100, 150, 200, 250, 300, 400, 500]:
        for n_n in [15, 30, 45]:
            run_name = f"ApproachB_HDBSCAN_mcs{mcs}_n{n_n}"
            try:
                b_model = StrategyBERTopic(vectorizer_type="transition", n_neighbors=n_n, min_cluster_size=mcs)
                b_model.fit(all_win_inputs)
                b_ids, _ = b_model.transform(all_win_inputs)
                n_b_topics = b_model.n_topics

                session_props = []
                frame_strat_dict = {}
                win_offset = 0
                for s_idx, s_id in enumerate(session_id_list):
                    n_win = n_windows_per_session[s_idx]
                    if n_win == 0:
                        session_props.append(np.zeros(max(1, n_b_topics)))
                        continue
                    w_ids = b_ids[win_offset : win_offset + n_win]
                    win_offset += n_win
                    counts = pd.Series(w_ids).value_counts()
                    p = [counts.get(t, 0) / n_win for t in range(n_b_topics)]
                    session_props.append(p)
                    frame_strat_dict[s_id] = np.repeat(w_ids, 800)

                X_props = np.array(session_props)
                acc, f1 = evaluate_classification(X_props, labels)
                k_act, h_norm, k_eff = compute_entropy_and_keff(X_props)
                kin_metrics = evaluate_session_kinematic_encoding(session_dfs, frame_strat_dict)

                results.append({
                    "approach": "Approach B (HDBSCAN Scale)",
                    "run_name": run_name,
                    "target_k": n_b_topics,
                    "active_k": k_act,
                    "group_decodability_acc_pct": acc * 100.0,
                    "weighted_f1": f1,
                    "entropy_norm": h_norm,
                    "k_effective": k_eff,
                    "eta2_speed": kin_metrics["eta2_velocity"],
                    "eta2_dist_center": kin_metrics["eta2_dist_center"],
                    "eta2_angular_vel": kin_metrics["eta2_angular_vel"],
                    "eta2_cadence": kin_metrics["eta2_cadence"]
                })
                print(f"  [B] mcs={mcs:3d} | n_neigh={n_n:2d} --> Active K: {k_act:2d} | Acc: {acc*100:5.2f}% | eta2_speed: {kin_metrics['eta2_velocity']:.3f} | eta2_dist: {kin_metrics['eta2_dist_center']:.3f}")
            except Exception as exc:
                print(f"  [B] Error in {run_name}: {exc}")

    # ---------------------------------------------------------
    # APPROACH C: Direct Fixed-K Mixture Models (9 Runs)
    # ---------------------------------------------------------
    print("\n--- Running Approach C: Direct Fixed-K GMM / LDA ---")
    for k_fixed in [4, 6, 8, 10, 12]:
        run_name_gmm = f"ApproachC_GMM_K{k_fixed}"
        try:
            gmm = GaussianMixture(n_components=k_fixed, random_state=42, covariance_type="diag")
            win_probs_gmm = gmm.fit_predict(X_vec)

            session_props = []
            frame_strat_dict = {}
            win_offset = 0
            for s_idx, s_id in enumerate(session_id_list):
                n_win = n_windows_per_session[s_idx]
                if n_win == 0:
                    session_props.append(np.zeros(k_fixed))
                    continue
                w_ids = win_probs_gmm[win_offset : win_offset + n_win]
                win_offset += n_win
                counts = pd.Series(w_ids).value_counts()
                p = [counts.get(t, 0) / n_win for t in range(k_fixed)]
                session_props.append(p)
                frame_strat_dict[s_id] = np.repeat(w_ids, 800)

            X_props = np.array(session_props)
            acc, f1 = evaluate_classification(X_props, labels)
            k_act, h_norm, k_eff = compute_entropy_and_keff(X_props)
            kin_metrics = evaluate_session_kinematic_encoding(session_dfs, frame_strat_dict)

            results.append({
                "approach": "Approach C (Fixed-K GMM)",
                "run_name": run_name_gmm,
                "target_k": k_fixed,
                "active_k": k_act,
                "group_decodability_acc_pct": acc * 100.0,
                "weighted_f1": f1,
                "entropy_norm": h_norm,
                "k_effective": k_eff,
                "eta2_speed": kin_metrics["eta2_velocity"],
                "eta2_dist_center": kin_metrics["eta2_dist_center"],
                "eta2_angular_vel": kin_metrics["eta2_angular_vel"],
                "eta2_cadence": kin_metrics["eta2_cadence"]
            })
            print(f"  [C-GMM] K={k_fixed:2d} --> Acc: {acc*100:5.2f}% | eta2_speed: {kin_metrics['eta2_velocity']:.3f} | eta2_dist: {kin_metrics['eta2_dist_center']:.3f}")
        except Exception as exc:
            print(f"  [C-GMM] Error in {run_name_gmm}: {exc}")

    df_res = pd.DataFrame(results)
    master_csv = os.path.join(out_dir, "broad_sweep_kinematic_benchmark.csv")
    df_res.to_csv(master_csv, index=False)
    print(f"\nSaved Broad Sweep Kinematic Benchmark to: {master_csv}")

    # ---------------------------------------------------------
    # CORRELATION ANALYSIS
    # ---------------------------------------------------------
    acc_arr = df_res["group_decodability_acc_pct"].values
    speed_arr = df_res["eta2_speed"].values
    dist_arr = df_res["eta2_dist_center"].values
    k_arr = df_res["active_k"].values

    r_speed, p_speed = pearsonr(acc_arr, speed_arr)
    r_dist, p_dist = pearsonr(acc_arr, dist_arr)
    rho_speed, _ = spearmanr(acc_arr, speed_arr)
    rho_dist, _ = spearmanr(acc_arr, dist_arr)

    r_k_acc, _ = pearsonr(k_arr, acc_arr)
    r_k_speed, _ = pearsonr(k_arr, speed_arr)

    print("\n" + "="*80)
    print("=== CORRELATION ANALYSIS ACROSS 45 BROAD SWEEP MODELS ===")
    print("="*80)
    print("1. Group Decodability Acc vs Locomotion Speed Encoding (eta2_speed):")
    print(f"   - Pearson Correlation  r = {r_speed:+.3f} (p = {p_speed:.4e})")
    print(f"   - Spearman Correlation r = {rho_speed:+.3f}")
    print("\n2. Group Decodability Acc vs Thigmotaxis Encoding (eta2_dist_center):")
    print(f"   - Pearson Correlation  r = {r_dist:+.3f} (p = {p_dist:.4e})")
    print(f"   - Spearman Correlation r = {rho_dist:+.3f}")
    print("\n3. Number of Active Vibes (K) Correlations:")
    print(f"   - Active K vs Decodability Acc: r = {r_k_acc:+.3f}")
    print(f"   - Active K vs Speed Encoding:   r = {r_k_speed:+.3f}")
    print("="*80)

if __name__ == "__main__":
    main()
