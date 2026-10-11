import os
import pickle
import sys

import numpy as np
import pandas as pd

# Path setup
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
    sys.path.append(os.path.join(ROOT_DIR, "syllable_generation"))

from studies.behavior.language_models.lib.data.dataset import SessionDataset
from studies.behavior.language_models.lib.evaluation.kinematic_encoding import evaluate_session_kinematic_encoding
from studies.behavior.language_models.lib.models.strategies import StrategyBERTopic
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
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

def evaluate_decodability(X_features: np.ndarray, y_labels: np.ndarray):
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

def apply_arousal_weighting(win_inputs: list) -> list:
    arousal_inputs = []
    for doc in win_inputs:
        if isinstance(doc, list):
            arr = np.array(doc)
            if len(arr) > 1:
                diff = np.diff(arr) != 0
                active_tokens = arr[:-1][diff].tolist()
                arousal_inputs.append(doc + active_tokens)
            else:
                arousal_inputs.append(doc)
        else:
            arousal_inputs.append(doc)
    return arousal_inputs

def main():
    out_dir = os.path.join(ROOT_DIR, "syllable_generation/analysis/language_models/output/strategies_extended")
    cache_path = os.path.join(out_dir, "cache/prep_amph_postural_gmm_k3_transition.pkl")

    print("=== Running Comprehensive Master Hyperparameter Benchmark & Correlation Analysis ===")

    dataset = SessionDataset(cache_path=cache_path)
    base_win_inputs = dataset.get_window_inputs()
    labels = dataset.get_cohort_labels()
    n_win_per_session = dataset.n_windows_per_session

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

    vocab_size = 300
    master_results = []

    # Define Comprehensive Model Grid
    sweep_grid = [
        # Axis 1: Baseline Sweep (mcs 150 vs 300)
        {"name": "Baseline_mcs150", "mcs": 150, "sigma": 0.0, "ngram": 1, "use_idf": False, "use_arousal": False},
        {"name": "Baseline_mcs300 (Sweetspot)", "mcs": 300, "sigma": 0.0, "ngram": 1, "use_idf": False, "use_arousal": False},

        # Axis 2: Postural Kernel Sigma Sweep
        {"name": "PosturalKernel_sigma0.25_mcs150", "mcs": 150, "sigma": 0.25, "ngram": 1, "use_idf": False, "use_arousal": False},
        {"name": "PosturalKernel_sigma0.50_mcs150", "mcs": 150, "sigma": 0.50, "ngram": 1, "use_idf": False, "use_arousal": False},
        {"name": "PosturalKernel_sigma1.00_mcs150", "mcs": 150, "sigma": 1.00, "ngram": 1, "use_idf": False, "use_arousal": False},
        {"name": "PosturalKernel_sigma0.25_mcs300", "mcs": 300, "sigma": 0.25, "ngram": 1, "use_idf": False, "use_arousal": False},
        {"name": "PosturalKernel_sigma0.50_mcs300", "mcs": 300, "sigma": 0.50, "ngram": 1, "use_idf": False, "use_arousal": False},

        # Axis 3: N-Gram Depth Sweep
        {"name": "NGram_2step_mcs150", "mcs": 150, "sigma": 0.0, "ngram": 2, "use_idf": False, "use_arousal": False},
        {"name": "NGram_2step_mcs300", "mcs": 300, "sigma": 0.0, "ngram": 2, "use_idf": False, "use_arousal": False},

        # Axis 4: LM Science Factorial Sweep (TF-IDF & Arousal Pooling)
        {"name": "Factorial_TFIDFOn_ArousalOff_mcs150", "mcs": 150, "sigma": 0.0, "ngram": 1, "use_idf": True, "use_arousal": False},
        {"name": "Factorial_TFIDFOff_ArousalOn_mcs150", "mcs": 150, "sigma": 0.0, "ngram": 1, "use_idf": False, "use_arousal": True},
        {"name": "Factorial_TFIDFOn_ArousalOn_mcs150", "mcs": 150, "sigma": 0.0, "ngram": 1, "use_idf": True, "use_arousal": True},
        {"name": "Factorial_TFIDFOn_ArousalOff_mcs300", "mcs": 300, "sigma": 0.0, "ngram": 1, "use_idf": True, "use_arousal": False},
        {"name": "Factorial_TFIDFOff_ArousalOn_mcs300", "mcs": 300, "sigma": 0.0, "ngram": 1, "use_idf": False, "use_arousal": True},
        {"name": "Factorial_TFIDFOn_ArousalOn_mcs300", "mcs": 300, "sigma": 0.0, "ngram": 1, "use_idf": True, "use_arousal": True},
    ]

    for cfg in sweep_grid:
        run_name = cfg["name"]
        mcs = cfg["mcs"]
        sigma = cfg["sigma"]
        ngram = cfg["ngram"]
        use_idf = cfg["use_idf"]
        use_arousal = cfg["use_arousal"]

        vec_type = "transition_2step" if ngram == 2 else "transition"
        win_inputs = apply_arousal_weighting(base_win_inputs) if use_arousal else base_win_inputs

        model = StrategyBERTopic(
            vectorizer_type=vec_type,
            n_neighbors=30,
            min_cluster_size=mcs,
            random_state=42,
            vocab_size=vocab_size,
            postural_kernel_sigma=sigma,
            ngram_depth=ngram,
            use_idf=use_idf
        )

        try:
            model.fit(win_inputs)
            strat_ids, _ = model.transform(win_inputs)

            n_topics = model.n_topics
            n_sessions = len(labels)
            props_matrix = np.zeros((n_sessions, n_topics))

            frame_strat_dict = {}
            win_offset = 0
            for s_idx, s_id in enumerate(session_id_list):
                n_win = n_win_per_session[s_idx]
                s_wins = strat_ids[win_offset : win_offset + n_win]
                win_offset += n_win

                counts = pd.Series(s_wins).value_counts(normalize=True)
                for t_id, prop in counts.items():
                    if 0 <= t_id < n_topics:
                        props_matrix[s_idx, t_id] = prop

                frame_strat_dict[s_id] = np.repeat(s_wins, 800)

            K_active, H_norm, K_eff = compute_entropy_and_keff(props_matrix)
            acc, f1 = evaluate_decodability(props_matrix, np.array(labels))
            kin_metrics = evaluate_session_kinematic_encoding(session_dfs, frame_strat_dict)

            print(f"Run: {run_name:<38} | Acc: {acc*100:5.2f}% | K={K_active:2d} | eta2_speed: {kin_metrics['eta2_velocity']:.3f} | eta2_dist: {kin_metrics['eta2_dist_center']:.3f}")

            master_results.append({
                "model_name": run_name,
                "min_cluster_size": mcs,
                "postural_sigma": sigma,
                "ngram_depth": ngram,
                "use_idf": use_idf,
                "use_arousal": use_arousal,
                "active_vibes_K": K_active,
                "normalized_entropy_Hnorm": H_norm,
                "group_decodability_acc_pct": acc * 100.0,
                "weighted_f1": f1,
                "eta2_speed": kin_metrics["eta2_velocity"],
                "eta2_dist_center": kin_metrics["eta2_dist_center"],
                "eta2_angular_vel": kin_metrics["eta2_angular_vel"],
                "eta2_cadence": kin_metrics["eta2_cadence"]
            })
        except Exception as e:
            print(f"❌ Failed run {run_name}: {e}")

    df_master = pd.DataFrame(master_results)
    csv_path = os.path.join(out_dir, "master_sweep_benchmark_table.csv")
    df_master.to_csv(csv_path, index=False)
    print(f"\nSaved Master Sweep Benchmark Table to: {csv_path}")

    # Compute Explicit Correlation between Group Discernibility (Decodability Acc) and Kinematic Encoding (eta2_speed, eta2_dist)
    acc_arr = df_master["group_decodability_acc_pct"].values
    speed_arr = df_master["eta2_speed"].values
    dist_arr = df_master["eta2_dist_center"].values

    r_speed, p_speed = pearsonr(acc_arr, speed_arr)
    r_dist, p_dist = pearsonr(acc_arr, dist_arr)

    rho_speed, _ = spearmanr(acc_arr, speed_arr)
    rho_dist, _ = spearmanr(acc_arr, dist_arr)

    print("\n" + "="*80)
    print("=== CORRELATION ANALYSIS: GROUP DISCERNIBILITY vs KINEMATIC ENCODING ===")
    print("="*80)
    print("1. Group Decodability Acc vs Locomotion Speed Encoding (eta2_speed):")
    print(f"   - Pearson Correlation  r = {r_speed:+.3f} (p = {p_speed:.4f})")
    print(f"   - Spearman Correlation r = {rho_speed:+.3f}")
    print("\n2. Group Decodability Acc vs Thigmotaxis Encoding (eta2_dist_center):")
    print(f"   - Pearson Correlation  r = {r_dist:+.3f} (p = {p_dist:.4f})")
    print(f"   - Spearman Correlation r = {rho_dist:+.3f}")
    print("="*80)

if __name__ == "__main__":
    main()
