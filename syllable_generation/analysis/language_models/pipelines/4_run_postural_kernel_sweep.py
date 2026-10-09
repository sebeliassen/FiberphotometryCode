import os
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

def build_postural_kernel_matrix(vocab_size: int, sigma: float) -> np.ndarray:
    """
    Builds a Gaussian similarity kernel matrix K[j, k] across vocabulary tokens.
    """
    if sigma <= 0.0:
        return np.eye(vocab_size, dtype=np.float32)

    # Distance matrix based on sub-state token indexing space
    indices = np.arange(vocab_size)
    dists = np.abs(indices[:, None] - indices[None, :]) / float(vocab_size)

    kernel = np.exp(-(dists ** 2) / (2.0 * (sigma ** 2)))
    # Row normalize kernel
    row_sums = kernel.sum(axis=1, keepdims=True)
    kernel = np.divide(kernel, row_sums, out=np.zeros_like(kernel), where=row_sums > 0)
    return kernel.astype(np.float32)

def main():
    log_dir = os.path.join(ROOT_DIR, "syllable_generation/analysis/language_models/output/strategies_extended/logs")
    out_dir = os.path.join(ROOT_DIR, "syllable_generation/analysis/language_models/output/strategies_extended")
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, "postural_kernel_sweep.log")

    def log_msg(msg: str):
        print(msg)
        with open(log_file, "a") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")

    log_msg("=== Starting 24-Run Postural Kernel & N-Gram Depth Comparative Sweep (Axis 4) ===")
    send_notification("🚀 Starting 24-Run Postural Kernel & N-Gram Depth Sweep (Axis 4)")

    cache_path = os.path.join(out_dir, "cache/prep_amph_postural_gmm_k3_transition.pkl")
    log_msg(f"Loading dataset representation from: {cache_path}")

    dataset = SessionDataset(cache_path=cache_path)
    all_win_inputs = dataset.get_window_inputs()
    labels = dataset.get_cohort_labels()
    n_win_per_session = dataset.n_windows_per_session

    log_msg(f"Dataset ready: {len(all_win_inputs)} sliding windows across {len(labels)} sessions.")

    vocab_size = 300
    results = []

    # Sweep Grid Definition
    sigmas = [0.0, 0.25, 0.50, 1.00]
    ngrams = [1, 2, 3]
    min_cluster_sizes = [150, 300]

    total_runs = len(sigmas) * len(ngrams) * len(min_cluster_sizes)
    run_count = 0

    start_time = time.time()

    for mcs in min_cluster_sizes:
        for N in ngrams:
            for sigma in sigmas:
                run_count += 1
                run_name = f"PosturalKernel_mcs{mcs}_N{N}_sigma{sigma:.2f}"
                log_msg(f"[{run_count}/{total_runs}] Running {run_name}...")

                kernel = build_postural_kernel_matrix(vocab_size, sigma)

                model = StrategyBERTopic(
                    vectorizer_type="transition",
                    n_neighbors=30,
                    min_cluster_size=mcs,
                    random_state=42,
                    vocab_size=vocab_size,
                    postural_kernel_sigma=sigma,
                    kernel_matrix=kernel if sigma > 0 else None,
                    ngram_depth=N
                )

                try:
                    model.fit(all_win_inputs)
                    strat_ids, _ = model.transform(all_win_inputs)

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
                        "ngram_depth": N,
                        "sigma": sigma,
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

    csv_out = os.path.join(out_dir, "postural_kernel_sweep_results.csv")
    df_res.to_csv(csv_out, index=False)
    log_msg(f"\nSaved full sweep results to: {csv_out}")

    best_row = df_res.sort_values(by="decodability_acc_pct", ascending=False).iloc[0]
    best_msg = f"🏆 Best Model: {best_row['run_name']} | Acc: {best_row['decodability_acc_pct']:.2f}% | K={best_row['active_vibes_K']} | H_norm={best_row['normalized_entropy_Hnorm']:.3f}"
    log_msg(best_msg)

    send_notification(f"✅ Axis 4 Sweep Complete ({elapsed_sec/60.0:.1f}m)! {best_msg}")
    log_msg("=== Postural Kernel Sweep Completed Successfully ===")

if __name__ == "__main__":
    main()
