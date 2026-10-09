import os
import sys
import time
import pickle
import urllib.request
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

# Paths & Settings
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
    sys.path.append(os.path.join(ROOT_DIR, "syllable_generation"))

from analysis.language_models.lib.models.strategies import StrategyBERTopic

NTFY_URL = "https://ntfy.sh/woody_videos_seb"

def send_notification(msg: str):
    try:
        req = urllib.request.Request(NTFY_URL, data=msg.encode('utf-8'))
        with urllib.request.urlopen(req) as resp:
            pass
    except Exception as e:
        print(f"Failed ntfy notification: {e}")

def compute_entropy_and_keff(proportions_matrix: np.ndarray):
    """
    Computes mean Normalized Shannon Entropy H_norm and Effective Clusters K_eff.
    proportions_matrix shape: (N_sessions, K_clusters)
    """
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
    """
    Runs 5-fold Stratified CV Random Forest classification.
    """
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
    print("=== Starting 45-Run Strategy Scale & Entropy Comparative Sweep ===")
    send_notification("🚀 Starting 45-Run Strategy Scale & Entropy Sweep (Approach A, B, C)")
    
    cache_path = os.path.join(ROOT_DIR, "syllable_generation/analysis/language_models/output/strategies_extended/cache/prep_amph_postural_gmm_k3_transition.pkl")
    print(f"Loading pre-cached dataset representation from: {cache_path}")
    
    with open(cache_path, "rb") as f:
        clusterer, all_win_inputs, session_window_meta = pickle.load(f)
        
    labels = np.array([meta[5] for meta in session_window_meta])
    n_windows_per_session = [meta[0] for meta in session_window_meta]
    
    print(f"Dataset ready: {len(all_win_inputs)} total sliding windows across {len(labels)} sessions.")
    
    results = []
    
    # ---------------------------------------------------------
    # APPROACH A: 2-Tier Hierarchical Merging (15 Runs)
    # ---------------------------------------------------------
    print("\n--- Running Approach A: 2-Tier Hierarchical Merging ---")
    base_model = StrategyBERTopic(vectorizer_type="transition", n_neighbors=30, min_cluster_size=75)
    base_model.fit(all_win_inputs)
    base_strat_ids, _ = base_model.transform(all_win_inputs)
    n_base_topics = base_model.n_topics
    
    # Extract numerical feature matrix X_vec
    X_vec = base_model._prepare_vectors(all_win_inputs)
    
    # Calculate centroid vector for each base topic
    base_topic_vectors = []
    for t in range(n_base_topics):
        mask = (base_strat_ids == t)
        if np.any(mask):
            base_topic_vectors.append(X_vec[mask].mean(axis=0))
        else:
            base_topic_vectors.append(np.zeros(X_vec.shape[1]))
    base_topic_vectors = np.array(base_topic_vectors)
    
    # Add tiny epsilon to prevent cosine distance divide-by-zero on unassigned topic centroids
    base_topic_vectors = base_topic_vectors + 1e-9
    
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
                
                # Session proportions
                session_props = []
                strat_idx = 0
                for n_win in n_windows_per_session:
                    if n_win == 0:
                        session_props.append(np.zeros(k_super))
                        continue
                    w_ids = super_win_ids[strat_idx : strat_idx + n_win]
                    strat_idx += n_win
                    counts = pd.Series(w_ids).value_counts()
                    p = [counts.get(t, 0) / n_win for t in range(k_super)]
                    session_props.append(p)
                X_props = np.array(session_props)
                
                acc, f1 = evaluate_classification(X_props, labels)
                k_act, h_norm, k_eff = compute_entropy_and_keff(X_props)
                
                results.append({
                    "approach": "Approach A (Hierarchy)",
                    "run_name": run_name,
                    "target_k": k_super,
                    "active_k": k_act,
                    "accuracy": acc,
                    "f1_score": f1,
                    "entropy_norm": h_norm,
                    "k_effective": k_eff,
                    "params": f"linkage={metric}, k={k_super}"
                })
                print(f"  [A] K={k_super:2d} | Metric={metric:9s} --> Acc: {acc*100:5.2f}% | H_norm: {h_norm:.3f} | K_eff: {k_eff:.2f}")
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
                strat_idx = 0
                for n_win in n_windows_per_session:
                    if n_win == 0:
                        session_props.append(np.zeros(max(1, n_b_topics)))
                        continue
                    w_ids = b_ids[strat_idx : strat_idx + n_win]
                    strat_idx += n_win
                    counts = pd.Series(w_ids).value_counts()
                    p = [counts.get(t, 0) / n_win for t in range(n_b_topics)]
                    session_props.append(p)
                X_props = np.array(session_props)
                
                acc, f1 = evaluate_classification(X_props, labels)
                k_act, h_norm, k_eff = compute_entropy_and_keff(X_props)
                
                results.append({
                    "approach": "Approach B (HDBSCAN Scale)",
                    "run_name": run_name,
                    "target_k": n_b_topics,
                    "active_k": k_act,
                    "accuracy": acc,
                    "f1_score": f1,
                    "entropy_norm": h_norm,
                    "k_effective": k_eff,
                    "params": f"mcs={mcs}, n_neigh={n_n}"
                })
                print(f"  [B] mcs={mcs:3d} | n_neigh={n_n:2d} --> Active K: {k_act:2d} | Acc: {acc*100:5.2f}% | H_norm: {h_norm:.3f} | K_eff: {k_eff:.2f}")
            except Exception as exc:
                print(f"  [B] Error in {run_name}: {exc}")

    # ---------------------------------------------------------
    # APPROACH C: Direct Fixed-K Mixture Models (10 Runs)
    # ---------------------------------------------------------
    print("\n--- Running Approach C: Direct Fixed-K GMM / LDA ---")
    for k_fixed in [4, 6, 8, 10, 12]:
        # C1: GMM
        run_name_gmm = f"ApproachC_GMM_K{k_fixed}"
        try:
            gmm = GaussianMixture(n_components=k_fixed, random_state=42, covariance_type="diag")
            win_probs_gmm = gmm.fit_predict(X_vec)
            
            session_props = []
            strat_idx = 0
            for n_win in n_windows_per_session:
                if n_win == 0:
                    session_props.append(np.zeros(k_fixed))
                    continue
                w_ids = win_probs_gmm[strat_idx : strat_idx + n_win]
                strat_idx += n_win
                counts = pd.Series(w_ids).value_counts()
                p = [counts.get(t, 0) / n_win for t in range(k_fixed)]
                session_props.append(p)
            X_props = np.array(session_props)
            
            acc, f1 = evaluate_classification(X_props, labels)
            k_act, h_norm, k_eff = compute_entropy_and_keff(X_props)
            
            results.append({
                "approach": "Approach C (Fixed-K GMM)",
                "run_name": run_name_gmm,
                "target_k": k_fixed,
                "active_k": k_act,
                "accuracy": acc,
                "f1_score": f1,
                "entropy_norm": h_norm,
                "k_effective": k_eff,
                "params": f"model=GMM, k={k_fixed}"
            })
            print(f"  [C-GMM] K={k_fixed:2d} --> Acc: {acc*100:5.2f}% | H_norm: {h_norm:.3f} | K_eff: {k_eff:.2f}")
        except Exception as exc:
            print(f"  [C-GMM] Error in {run_name_gmm}: {exc}")
            
        # C2: LDA
        run_name_lda = f"ApproachC_LDA_K{k_fixed}"
        try:
            lda = LatentDirichletAllocation(n_components=k_fixed, random_state=42, n_jobs=-1)
            win_probs_lda = lda.fit_transform(X_vec)
            
            session_props = []
            strat_idx = 0
            for n_win in n_windows_per_session:
                if n_win == 0:
                    session_props.append(np.zeros(k_fixed))
                    continue
                w_p = win_probs_lda[strat_idx : strat_idx + n_win]
                strat_idx += n_win
                session_props.append(np.mean(w_p, axis=0))
            X_props = np.array(session_props)
            
            acc, f1 = evaluate_classification(X_props, labels)
            k_act, h_norm, k_eff = compute_entropy_and_keff(X_props)
            
            results.append({
                "approach": "Approach C (Fixed-K LDA)",
                "run_name": run_name_lda,
                "target_k": k_fixed,
                "active_k": k_act,
                "accuracy": acc,
                "f1_score": f1,
                "entropy_norm": h_norm,
                "k_effective": k_eff,
                "params": f"model=LDA, k={k_fixed}"
            })
            print(f"  [C-LDA] K={k_fixed:2d} --> Acc: {acc*100:5.2f}% | H_norm: {h_norm:.3f} | K_eff: {k_eff:.2f}")
        except Exception as exc:
            print(f"  [C-LDA] Error in {run_name_lda}: {exc}")

    # ---------------------------------------------------------
    # EXPORT RESULTS & VISUALIZATIONS
    # ---------------------------------------------------------
    df_res = pd.DataFrame(results).sort_values(by="accuracy", ascending=False)
    
    out_dir = os.path.join(ROOT_DIR, "syllable_generation/analysis/language_models/output/strategies_extended")
    os.makedirs(out_dir, exist_ok=True)
    
    report_path = os.path.join(out_dir, "vibe_scale_sweep_results.md")
    
    with open(report_path, "w") as rf:
        rf.write("# Strategy Scale & Entropy Comparative Sweep Results\n\n")
        rf.write(f"Total evaluated configurations: **{len(df_res)}**\n\n")
        rf.write("| Rank | Approach | Config Name | Active Categories ($K$) | Effective $K_{\\text{eff}}$ | Entropy Balance ($H_{\\text{norm}}$) | Classifier Accuracy |\n")
        rf.write("|---|---|---|---|---|---|---|\n")
        for rank, row in enumerate(df_res.itertuples(), 1):
            rf.write(f"| {rank} | {row.approach} | `{row.run_name}` | {row.active_k} | {row.k_effective:.2f} | **{row.entropy_norm:.3f}** | **{row.accuracy*100:.2f}%** |\n")
            
    print(f"\nSaved Markdown Results to: {report_path}")
    
    # Generate scatter plot: Accuracy vs Category Count (K) colored by H_norm
    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(
        df_res["active_k"], df_res["accuracy"] * 100,
        c=df_res["entropy_norm"], cmap="viridis", s=100, edgecolors="black", alpha=0.9
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Normalized Strategy Entropy ($H_{norm}$)", fontsize=11)
    
    ax.set_xlabel("Number of Active Macro-Strategies ($K$)", fontsize=12)
    ax.set_ylabel("Classifier Accuracy (%)", fontsize=12)
    ax.set_title("Strategy Scale vs. Decodability & Entropy Balance", fontsize=14, pad=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    
    plot_path = os.path.join(out_dir, "vibe_scale_sweep_plot.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Visualization Plot to: {plot_path}")
    
    top_row = df_res.iloc[0]
    msg = f"🏆 Strategy Scale Sweep Completed! Total: {len(df_res)} runs. Top Model: {top_row['run_name']} (Acc: {top_row['accuracy']*100:.2f}%, K={top_row['active_k']}, H_norm={top_row['entropy_norm']:.3f})"
    send_notification(msg)
    print(f"\n{msg}")

if __name__ == "__main__":
    main()
