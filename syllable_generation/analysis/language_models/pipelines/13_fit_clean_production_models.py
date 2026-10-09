"""
Fit Clean Production Strategy Models (9-Vibe & 20-Vibe Models)
Located at: syllable_generation/analysis/language_models/pipelines/13_fit_clean_production_models.py

This script:
1. Loads clean window inputs (filtered pre-entry frames) from prep_amph_postural_gmm_k3_transition.pkl.
2. Fits the production 9-Vibe Model (mcs=300, nn=30) and 20-Vibe Model (mcs=150, nn=30).
3. Evaluates Random Forest cohort decodability accuracy for both.
4. Exports clean session_strategy_intervals.csv and fitted_9vibe_model.pkl.
"""

import os
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
    sys.path.append(os.path.join(ROOT_DIR, "syllable_generation"))

from analysis.language_models.lib.data.dataset import SessionDataset
from analysis.language_models.lib.models.strategies import StrategyBERTopic

OUT_DIR = os.path.join(ROOT_DIR, "syllable_generation/analysis/language_models/output/strategies_extended")

def evaluate_decodability(X_features: np.ndarray, y_labels: np.ndarray) -> float:
    unique_classes, counts = np.unique(y_labels, return_counts=True)
    n_splits = min(5, np.min(counts))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs = []
    for train_idx, test_idx in skf.split(X_features, y_labels):
        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        clf.fit(X_features[train_idx], y_labels[train_idx])
        accs.append(accuracy_score(y_labels[test_idx], clf.predict(X_features[test_idx])))
    return float(np.mean(accs))

def export_intervals(model, dataset, out_csv_path: str):
    all_win_inputs = dataset.get_window_inputs()
    labels = dataset.get_cohort_labels()
    session_window_meta = dataset.session_window_meta
    n_windows_per_session = dataset.n_windows_per_session

    strat_ids, _ = model.transform(all_win_inputs)
    n_topics = model.n_topics

    interval_rows = []
    win_offset = 0

    # 80.025 Hz tracking sampling rate, stride 800 frames = 9.996875s
    stride_sec = 800.0 / 80.025
    win_dur_sec = 10240.0 / 80.025

    for s_idx, meta in enumerate(session_window_meta):
        n_win = meta[0]
        abs_start_frame = meta[1]
        session_id = meta[3]
        vid_path = meta[4]

        w_ids = strat_ids[win_offset : win_offset + n_win]
        win_offset += n_win

        # Run-length encode strategy window sequence
        if len(w_ids) == 0:
            continue

        current_id = w_ids[0]
        start_w = 0

        for w_i in range(1, len(w_ids)):
            if w_ids[w_i] != current_id:
                s_frame = abs_start_frame + int(start_w * 800)
                e_frame = abs_start_frame + int(w_i * 800) + 10240
                s_time = start_w * stride_sec
                e_time = w_i * stride_sec + win_dur_sec

                interval_rows.append({
                    "session_id": session_id,
                    "video_path": vid_path,
                    "strategy_id": int(current_id),
                    "start_frame": s_frame,
                    "end_frame": e_frame,
                    "start_time_sec": s_time,
                    "end_time_sec": e_time,
                })
                current_id = w_ids[w_i]
                start_w = w_i

        # Last interval
        s_frame = abs_start_frame + int(start_w * 800)
        e_frame = abs_start_frame + int(len(w_ids) * 800) + 10240
        s_time = start_w * stride_sec
        e_time = (len(w_ids) - 1) * stride_sec + win_dur_sec
        interval_rows.append({
            "session_id": session_id,
            "video_path": vid_path,
            "strategy_id": int(current_id),
            "start_frame": s_frame,
            "end_frame": e_frame,
            "start_time_sec": s_time,
            "end_time_sec": e_time,
        })

    df_int = pd.DataFrame(interval_rows)
    df_int.to_csv(out_csv_path, index=False)
    print(f"Exported {len(df_int)} clean strategy intervals to: {out_csv_path}")
    return df_int

def main():
    print("=" * 75)
    print("FITTING CLEAN PRODUCTION STRATEGY MODELS (9-Vibe & 20-Vibe)")
    print("=" * 75)

    cache_path = os.path.join(OUT_DIR, "cache/prep_amph_postural_gmm_k3_transition.pkl")
    dataset = SessionDataset(cache_path=cache_path)
    all_win_inputs = dataset.get_window_inputs()
    labels = dataset.get_cohort_labels()
    n_windows_per_session = dataset.n_windows_per_session

    print(f"Loaded clean dataset: {dataset.get_num_sessions()} sessions | {len(all_win_inputs)} total windows")

    # 1. Fit Fine-Grained Model (K=21, nn=11, mcs=163) -> Top Trial #294 (53.3% 4-Way Acc)
    print("\n--- Fitting Fine-Grained Strategy Model (mcs=163, n_neighbors=11) ---")
    model_20 = StrategyBERTopic(vectorizer_type="transition", n_neighbors=11, min_cluster_size=163, random_state=42)
    model_20.fit(all_win_inputs)
    strat_ids_20, _ = model_20.transform(all_win_inputs)
    n_topics_20 = model_20.n_topics

    props_20 = np.zeros((len(labels), max(1, n_topics_20)))
    win_offset = 0
    for s_idx, n_win in enumerate(n_windows_per_session):
        w_ids = strat_ids_20[win_offset : win_offset + n_win]
        win_offset += n_win
        counts = pd.Series(w_ids).value_counts(normalize=True)
        for t_id, prop in counts.items():
            if 0 <= t_id < n_topics_20:
                props_20[s_idx, t_id] = prop

    acc_20 = evaluate_decodability(props_20, np.array(labels))
    print(f"🏆 Fine-Grained Model: Discovered K={n_topics_20} strategies | 4-Way Cohort Accuracy = {acc_20*100:.2f}%")

    # 2. Fit Production 9-Vibe Model (K=9, nn=12, mcs=237) -> Top Trial #90 (50.0% 4-Way Acc, 86.7% GT Acc)
    print("\n--- Fitting Production 9-Vibe Model (mcs=237, n_neighbors=12) ---")
    model_9 = StrategyBERTopic(vectorizer_type="transition", n_neighbors=12, min_cluster_size=237, random_state=42)
    model_9.fit(all_win_inputs)
    strat_ids_9, _ = model_9.transform(all_win_inputs)
    n_topics_9 = model_9.n_topics

    props_9 = np.zeros((len(labels), max(1, n_topics_9)))
    win_offset = 0
    for s_idx, n_win in enumerate(n_windows_per_session):
        w_ids = strat_ids_9[win_offset : win_offset + n_win]
        win_offset += n_win
        counts = pd.Series(w_ids).value_counts(normalize=True)
        for t_id, prop in counts.items():
            if 0 <= t_id < n_topics_9:
                props_9[s_idx, t_id] = prop

    acc_9 = evaluate_decodability(props_9, np.array(labels))
    print(f"🏆 Production 9-Vibe Model: Discovered K={n_topics_9} strategies | 4-Way Cohort Accuracy = {acc_9*100:.2f}%")

    # Save 9-vibe model checkpoint
    model_pkl_path = os.path.join(OUT_DIR, "fitted_9vibe_model.pkl")
    with open(model_pkl_path, "wb") as f:
        pickle.dump(model_9, f)
    print(f"Saved fitted 9-vibe model checkpoint to: {model_pkl_path}")

    # Save 21-vibe model checkpoint
    model_20_pkl_path = os.path.join(OUT_DIR, "fitted_21vibe_model.pkl")
    with open(model_20_pkl_path, "wb") as f:
        pickle.dump(model_20, f)
    print(f"Saved fitted 21-vibe model checkpoint to: {model_20_pkl_path}")

    # Export clean strategy intervals
    intervals_csv_path = os.path.join(OUT_DIR, "session_strategy_intervals.csv")
    export_intervals(model_9, dataset, intervals_csv_path)

    print("\n" + "=" * 75)
    print("PRODUCTION MODELS FITTED & EXPORTED SUCCESSFULLY!")
    print("=" * 75)

if __name__ == "__main__":
    main()
