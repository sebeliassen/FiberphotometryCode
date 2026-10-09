import os
import sys
import time
import pickle
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

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
    """
    Applies kinetic arousal window pooling by boosting active non-zero transition weights.
    """
    arousal_inputs = []
    for doc in win_inputs:
        if isinstance(doc, list):
            # Duplicate non-repetitive active transitions to simulate kinetic arousal weighting
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
    
    log_file = os.path.join(log_dir, "lm_weighting_sweep.log")
    
    def log_msg(msg: str):
        print(msg)
        with open(log_file, "a") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")
            
    log_msg("=== Starting 8-Run LM Science Factorial Sweep (TF-IDF & Arousal Pooling) ===")
    send_notification("🚀 Starting 8-Run LM Science Factorial Sweep (TF-IDF & Arousal Pooling)")
    
    cache_path = os.path.join(out_dir, "cache/prep_amph_postural_gmm_k3_transition.pkl")
    log_msg(f"Loading dataset representation from: {cache_path}")
    
    dataset = SessionDataset(cache_path=cache_path)
    base_win_inputs = dataset.get_window_inputs()
    labels = dataset.get_cohort_labels()
    n_win_per_session = dataset.n_windows_per_session
    
    log_msg(f"Dataset ready: {len(base_win_inputs)} sliding windows across {len(labels)} sessions.")
    
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
                
                log_msg(f"[{run_count}/{total_runs}] Running {run_name}...")
                
                # Prepare window inputs (apply arousal frame weighting if enabled)
                if use_arousal:
                    win_inputs = apply_arousal_weighting(base_win_inputs)
                else:
                    win_inputs = base_win_inputs
                    
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
                    
                    log_msg(f"  --> Acc: {acc*100:.2f}%, F1: {f1:.3f}, K: {K_active}, K_eff: {K_eff:.2f}, H_norm: {H_norm:.3f}")
                    
                    results.append({
                        "run_name": run_name,
                        "min_cluster_size": mcs,
                        "use_idf": use_idf,
                        "use_arousal": use_arousal,
                        "active_vibes_K": K_active,
                        "effective_K_eff": K_eff,
                        "normalized_entropy_Hnorm": H_norm,
                        "decodability_acc_pct": acc * 100.0,
                        "weighted_f1": f1
                    })
                except Exception as run_err:
                    log_msg(f"  ❌ Run failed: {run_err}")
                    
    elapsed_sec = time.time() - start_time
    df_res = pd.DataFrame(results)
    
    csv_out = os.path.join(out_dir, "lm_weighting_sweep_results.csv")
    df_res.to_csv(csv_out, index=False)
    log_msg(f"\nSaved full 8-run factorial results to: {csv_out}")
    
    best_row = df_res.sort_values(by="decodability_acc_pct", ascending=False).iloc[0]
    best_msg = f"🏆 Best Model: {best_row['run_name']} | Acc: {best_row['decodability_acc_pct']:.2f}% | K={best_row['active_vibes_K']} | H_norm={best_row['normalized_entropy_Hnorm']:.3f}"
    log_msg(best_msg)
    
    send_notification(f"✅ 8-Run Factorial Sweep Complete ({elapsed_sec/60.0:.1f}m)! {best_msg}")
    log_msg("=== 8-Run LM Science Factorial Sweep Completed Successfully ===")

if __name__ == "__main__":
    main()
