import os
import pickle
import sys
import urllib.request

import numpy as np
import pandas as pd

# Path setup
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
    sys.path.append(os.path.join(ROOT_DIR, "syllable_generation"))

from analysis.language_models.lib.data.dataset import SessionDataset
from analysis.language_models.lib.models.strategies import StrategyBERTopic
from analysis.language_models.lib.preprocessing.segmentation import assign_frame_strategies
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

def export_strategy_intervals_from_meta(session_window_meta, strat_ids, output_csv_path):
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    rows = []
    strat_idx = 0
    for meta in session_window_meta:
        n_win, windows, fs, num_frames, video_path, group_label, session_id = meta
        if n_win == 0:
            continue

        s_ids = strat_ids[strat_idx : strat_idx + n_win]
        strat_idx += n_win

        win_df = pd.DataFrame({
            "start_frame": [w["start_frame"] for w in windows],
            "end_frame": [w["end_frame"] for w in windows],
            "strategy_id": s_ids
        })

        frame_strats = assign_frame_strategies(win_df, num_frames)

        i = 0
        n_frames = len(frame_strats)
        while i < n_frames:
            curr_strat = frame_strats[i]
            start_frame = i
            while i < n_frames and frame_strats[i] == curr_strat:
                i += 1
            end_frame = i

            rows.append({
                "session_id": session_id,
                "video_path": video_path,
                "strategy_id": int(curr_strat),
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start_time_sec": float(start_frame / fs),
                "end_time_sec": float(end_frame / fs)
            })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(output_csv_path, index=False)
    print(f"Exported {len(df_out)} strategy intervals to: {output_csv_path}")

def main():
    print("=== Pipeline 1: Fitting 9-Vibe Strategy Model & Exporting Intervals ===")
    send_notification("🚀 Running Pipeline 1: Fitting 9-Vibe Model & Interval Export")

    cache_path = os.path.join(ROOT_DIR, "syllable_generation/analysis/language_models/output/strategies_extended/cache/prep_amph_postural_gmm_k3_transition.pkl")
    dataset = SessionDataset(cache_path=cache_path)

    all_win_inputs = dataset.get_window_inputs()
    labels = dataset.get_cohort_labels()
    n_win_per_session = dataset.n_windows_per_session

    print(f"Loaded dataset: {len(all_win_inputs)} sliding windows across {len(labels)} sessions.")

    # Fit 9-Vibe Strategy Model (HDBSCAN mcs=300, n_neighbors=30)
    print("Fitting StrategyBERTopic(vectorizer_type='transition', n_neighbors=30, min_cluster_size=300)...")
    model = StrategyBERTopic(vectorizer_type="transition", n_neighbors=30, min_cluster_size=300, random_state=42)
    model.fit(all_win_inputs)
    strat_ids, _ = model.transform(all_win_inputs)

    # Calculate session-level strategy proportions
    n_topics = model.n_topics
    n_sessions = len(labels)
    props_matrix = np.zeros((n_sessions, n_topics))

    win_offset = 0
    for s_idx, n_win in enumerate(n_win_per_session):
        s_ids = strat_ids[win_offset : win_offset + n_win]
        win_offset += n_win
        counts = pd.Series(s_ids).value_counts(normalize=True)
        for t_id, prop in counts.items():
            if 0 <= t_id < n_topics:
                props_matrix[s_idx, t_id] = prop

    K_active, H_norm, K_eff = compute_entropy_and_keff(props_matrix)
    acc, f1 = evaluate_decodability(props_matrix, np.array(labels))

    print(f"Results -> Active Vibes (K): {K_active}, Effective Count (K_eff): {K_eff:.2f}, Normalized Entropy (H_norm): {H_norm:.3f}")
    print(f"Decodability Accuracy: {acc*100:.2f}%, F1 Score: {f1:.3f}")

    # Save checkpoint
    out_dir = os.path.join(ROOT_DIR, "syllable_generation/analysis/language_models/output/strategies_extended")
    os.makedirs(out_dir, exist_ok=True)

    checkpoint_path = os.path.join(out_dir, "fitted_9vibe_model.pkl")
    with open(checkpoint_path, "wb") as f:
        pickle.dump({"model": model, "props_matrix": props_matrix, "labels": labels}, f)
    print(f"Saved model checkpoint to: {checkpoint_path}")

    # Export interval CSV
    csv_path = os.path.join(out_dir, "session_strategy_intervals.csv")
    export_strategy_intervals_from_meta(
        session_window_meta=dataset.session_window_meta,
        strat_ids=strat_ids,
        output_csv_path=csv_path
    )

    send_notification(f"✅ Pipeline 1 Complete! K={K_active}, Acc={acc*100:.1f}%, H_norm={H_norm:.3f}")
    print("=== Pipeline 1 Completed Successfully ===")

if __name__ == "__main__":
    main()
