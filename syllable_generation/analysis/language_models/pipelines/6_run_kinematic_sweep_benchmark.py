import os
import pickle
import sys
import time
import urllib.request

import numpy as np
import pandas as pd

# Path setup
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
    sys.path.append(os.path.join(ROOT_DIR, "syllable_generation"))

from analysis.language_models.lib.data.dataset import SessionDataset
from analysis.language_models.lib.evaluation.kinematic_encoding import evaluate_session_kinematic_encoding
from analysis.language_models.lib.models.strategies import StrategyBERTopic
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

NTFY_URL = "https://ntfy.sh/woody_videos_seb"

def send_notification(msg: str):
    try:
        req = urllib.request.Request(NTFY_URL, data=msg.encode('utf-8'))
        with urllib.request.urlopen(req) as resp:
            pass
    except Exception as e:
        print(f"Failed ntfy notification: {e}")

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
    log_dir = os.path.join(ROOT_DIR, "syllable_generation/analysis/language_models/output/strategies_extended/logs")
    out_dir = os.path.join(ROOT_DIR, "syllable_generation/analysis/language_models/output/strategies_extended")
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, "kinematic_sweep_benchmark.log")

    def log_msg(msg: str):
        print(msg)
        with open(log_file, "a") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")

    log_msg("=== Starting Across-Sweep Kinematic Encoding Benchmark & Vibe Passports ===")
    send_notification("🚀 Starting Across-Sweep Kinematic Encoding Benchmark & Vibe Passports")

    cache_path = os.path.join(out_dir, "cache/prep_amph_postural_gmm_k3_transition.pkl")
    log_msg(f"Loading dataset representation from: {cache_path}")

    dataset = SessionDataset(cache_path=cache_path)
    base_win_inputs = dataset.get_window_inputs()
    labels = dataset.get_cohort_labels()
    n_win_per_session = dataset.n_windows_per_session

    # Load session dataframes for frame-level kinematic calculations
    pkl_path = os.path.join(ROOT_DIR, "syllable_generation/data/jonathan_amph_sessions_v2_w_sylls.pkl")
    if not os.path.exists(pkl_path):
        pkl_path = os.path.join(ROOT_DIR, "syllable_generation/data/jonathan_amph_sessions_w_sylls.pkl")

    log_msg(f"Loading raw session dataframes from: {pkl_path}")
    with open(pkl_path, "rb") as f:
        raw_sessions = pickle.load(f)

    session_dfs = {}
    session_id_list = []
    for meta_tuple in dataset.session_window_meta:
        # meta_tuple: (win_data, fs, num_frames, video_path, group_label, session_id, ...)
        s_id = meta_tuple[6]
        session_id_list.append(s_id)

    for s in raw_sessions:
        exact_id = s.session_id
        for s_id in session_id_list:
            if s_id.startswith(exact_id) or exact_id.startswith(s_id):
                session_dfs[s_id] = s.all_data_df
                break

    log_msg(f"Matched {len(session_dfs)} session dataframes for kinematic encoding evaluation.")

    vocab_size = 300
    results = []

    # 8-Run Factorial Grid: mcs x use_idf x use_arousal
    scales = [150, 300]
    idfs = [False, True]
    arousals = [False, True]

    total_runs = len(scales) * len(idfs) * len(arousals)
    run_count = 0
    start_time = time.time()

    for mcs in scales:
        for use_arousal in arousals:
            for use_idf in idfs:
                run_count += 1
                arousal_str = "ArousalOn" if use_arousal else "ArousalOff"
                idf_str = "TFIDFOn" if use_idf else "TFIDFOff"
                run_name = f"Factorial_mcs{mcs}_{idf_str}_{arousal_str}"

                log_msg(f"[{run_count}/{total_runs}] Evaluating {run_name}...")

                win_inputs = apply_arousal_weighting(base_win_inputs) if use_arousal else base_win_inputs

                model = StrategyBERTopic(
                    vectorizer_type="transition",
                    n_neighbors=30,
                    min_cluster_size=mcs,
                    random_state=42,
                    vocab_size=vocab_size,
                    use_idf=use_idf
                )

                try:
                    model.fit(win_inputs)
                    strat_ids, _ = model.transform(win_inputs)

                    n_topics = model.n_topics
                    n_sessions = len(labels)
                    props_matrix = np.zeros((n_sessions, n_topics))

                    # Frame-level strategy reconstruction
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

                        # Broadcast window strategies to frame level (10-second windows = 800 frames)
                        frame_strats = np.repeat(s_wins, 800)
                        frame_strat_dict[s_id] = frame_strats

                    K_active, H_norm, K_eff = compute_entropy_and_keff(props_matrix)
                    acc, f1 = evaluate_decodability(props_matrix, np.array(labels))

                    # Evaluate Kinematic Encoding Across 4 Continuous Axes
                    kin_metrics = evaluate_session_kinematic_encoding(session_dfs, frame_strat_dict)

                    log_msg(
                        f"  --> Acc: {acc*100:.2f}% | K={K_active} | "
                        f"eta2_speed: {kin_metrics['eta2_velocity']:.3f} | "
                        f"eta2_dist_center: {kin_metrics['eta2_dist_center']:.3f} | "
                        f"eta2_angular_vel: {kin_metrics['eta2_angular_vel']:.3f} | "
                        f"eta2_cadence: {kin_metrics['eta2_cadence']:.3f}"
                    )

                    results.append({
                        "run_name": run_name,
                        "min_cluster_size": mcs,
                        "use_idf": use_idf,
                        "use_arousal": use_arousal,
                        "active_vibes_K": K_active,
                        "effective_K_eff": K_eff,
                        "normalized_entropy_Hnorm": H_norm,
                        "decodability_acc_pct": acc * 100.0,
                        "weighted_f1": f1,
                        "eta2_speed": kin_metrics["eta2_velocity"],
                        "eta2_dist_center": kin_metrics["eta2_dist_center"],
                        "eta2_angular_vel": kin_metrics["eta2_angular_vel"],
                        "eta2_cadence": kin_metrics["eta2_cadence"]
                    })
                except Exception as run_err:
                    log_msg(f"  ❌ Run failed: {run_err}")

    elapsed_sec = time.time() - start_time
    df_res = pd.DataFrame(results)

    csv_out = os.path.join(out_dir, "lm_weighting_sweep_results.csv")
    df_res.to_csv(csv_out, index=False)
    log_msg(f"\nSaved updated sweep benchmark CSV with kinematic encoding columns to: {csv_out}")

    send_notification(f"✅ Across-Sweep Kinematic Encoding Benchmark Complete ({elapsed_sec/60.0:.1f}m)!")
    log_msg("=== Across-Sweep Kinematic Encoding Benchmark Completed Successfully ===")

if __name__ == "__main__":
    main()
