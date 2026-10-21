import os
import pickle
import sys
import time

import matplotlib
import numpy as np
import pandas as pd
import requests

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

import wandb

ROOT_DIR = "/home/fsp585/Desktop/FiberphotometryCode"
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "syllable_generation/analysis/language_models"))

from studies.behavior.language_models.lib.evaluation.scalar_probes import run_scalar_probing_battery
from studies.behavior.language_models.lib.models.benchmark_models import (
    AgglomerativeStrategyClusterer,
    BERTopicStrategyClusterer,
    GMMStrategyClusterer,
    KMeansStrategyClusterer,
    SpectralStrategyClusterer,
)
from studies.behavior.language_models.lib.models.continuous import StrategyConditionedPoseModel

from fpcode.common.notify import send_ntfy

def eval_rf_classifier(X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> float:
    if len(set(y)) < 2:
        return 0.0
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    y_preds = np.zeros(len(y), dtype=object)
    for train_idx, test_idx in skf.split(X, y):
        rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        rf.fit(X[train_idx], y[train_idx])
        y_preds[test_idx] = rf.predict(X[test_idx])
    return float(accuracy_score(y, y_preds))

def main():
    start_time = time.time()
    out_dir = os.path.join(ROOT_DIR, "syllable_generation/analysis/language_models/output/strategies_extended")
    os.makedirs(out_dir, exist_ok=True)
    log_file = os.path.join(out_dir, "logs/multi_clustering_benchmark.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    def log(msg: str):
        ts = time.strftime('%Y-%m-%d %H:%M:%S')
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        with open(log_file, "a") as f:
            f.write(line + "\n")

    log("="*70)
    log("Starting Systematic Multi-Clustering Resolution Benchmark")
    send_ntfy("Starting Systematic Multi-Clustering Resolution Benchmark across 5 Model Families", "Benchmark Started")

    # 1. Initialize Weights & Biases
    wandb_run = wandb.init(
        project="language_models_in_vivo",
        entity="sebeliassen",
        name="multi_clustering_resolution_benchmark",
        job_type="clustering_benchmark",
        config={
            "dataset": "jonathan_amph_clean",
            "benchmark_families": ["GMM", "KMeans", "Spectral", "Agglomerative", "BERTopic"],
            "k_grid": [3, 5, 8, 10, 13, 16, 20, 25, 30]
        }
    )

    # 2. Load Cached Data
    cache_path = os.path.join(out_dir, "continuous_bout_cache.pkl")
    with open(cache_path, "rb") as f:
        data = pickle.load(f)
    session_bouts = data["session_bouts"]

    window_scalars_cache = os.path.join(out_dir, "window_scalars_cache.pkl")
    with open(window_scalars_cache, "rb") as f:
        scalar_cache_data = pickle.load(f)
    all_win_scalars = scalar_cache_data["scalars"]
    all_raw_sylls = scalar_cache_data["raw_sylls"]
    all_trans_inputs = scalar_cache_data["trans_inputs"]
    win_metadata = scalar_cache_data["meta"]
    log(f"Loaded {len(all_trans_inputs)} windows and {sum(len(b) for b in session_bouts)} bouts.")

    # Flatten bouts
    all_bouts_list = [b for s_bouts in session_bouts for b in s_bouts]
    all_bout_vectors = np.array([b["vector"] for b in all_bouts_list])
    n_train_bouts = int(0.8 * len(all_bout_vectors))
    X_train_bouts, X_test_bouts = all_bout_vectors[:n_train_bouts], all_bout_vectors[n_train_bouts:]

    # Window targets & metadata
    scalar_targets = {
        "Spatial Exploration Entropy": np.array([s["spatial_entropy"] for s in all_win_scalars]),
        "Switch Cadence": np.array([s["cadence"] for s in all_win_scalars]),
        "Turning Chirality": np.array([s["chirality"] for s in all_win_scalars]),
        "Sequence Transition Entropy": np.array([s["sequence_entropy"] for s in all_win_scalars])
    }

    # Load session objects for cohort / injection metadata
    pkl_path = os.path.join(ROOT_DIR, "syllable_generation/data/jonathan_amph_sessions_v2_w_sylls.pkl")
    with open(pkl_path, "rb") as f:
        raw_sessions = pickle.load(f)
    session_lookup = {s.session_id: s for s in raw_sessions}

    # Session slicing metadata
    session_groups = {}
    for idx, w in enumerate(win_metadata):
        sid = w["session_id"]
        if sid not in session_groups:
            s_obj = session_lookup[sid]
            meta = getattr(s_obj, "metadata", {})
            gt = meta.get("genotype", "WT")
            drug = meta.get("drug", "Saline")
            inj_f = meta.get("injection_frame", 0)
            session_groups[sid] = {
                "group": f"{gt}_{drug}",
                "genotype": gt,
                "drug": drug,
                "inj_frame": inj_f,
                "window_indices": [],
                "post_indices": []
            }
        session_groups[sid]["window_indices"].append(idx)
        if w["mid_frame"] >= session_groups[sid]["inj_frame"]:
            session_groups[sid]["post_indices"].append(idx)

    session_ids_ordered = list(session_groups.keys())
    labels_gt = np.array([session_groups[sid]["genotype"] for sid in session_ids_ordered])
    labels_drug = np.array([session_groups[sid]["drug"] for sid in session_ids_ordered])

    # Midpoints for bout mapping
    win_mids = np.array([w["mid_frame"] for w in win_metadata])

    # 3. Define Dense Parameter Grid Across Families
    K_GRID = [3, 5, 8, 10, 13, 16, 20, 25, 30]
    configs = []

    # A. GMM (Diag and Full Covariance)
    for k in K_GRID:
        configs.append({"family": "GMM", "variant": "GMM (diag)", "k_target": k, "model": GMMStrategyClusterer(n_components=k, covariance_type="diag")})
        configs.append({"family": "GMM", "variant": "GMM (full)", "k_target": k, "model": GMMStrategyClusterer(n_components=k, covariance_type="full")})

    # B. KMeans (Euclidean & Spherical)
    for k in K_GRID:
        configs.append({"family": "KMeans", "variant": "KMeans (Euclidean)", "k_target": k, "model": KMeansStrategyClusterer(n_clusters=k, spherical=False)})
        configs.append({"family": "KMeans", "variant": "KMeans (Spherical)", "k_target": k, "model": KMeansStrategyClusterer(n_clusters=k, spherical=True)})

    # C. Agglomerative (Ward Linkage)
    for k in K_GRID:
        configs.append({"family": "Agglomerative", "variant": "Agglomerative (Ward)", "k_target": k, "model": AgglomerativeStrategyClusterer(n_clusters=k)})

    # D. Spectral Clustering (nn=15, 30)
    for k in [3, 5, 8, 10, 13, 16, 20]:
        configs.append({"family": "Spectral", "variant": "Spectral (nn=30)", "k_target": k, "model": SpectralStrategyClusterer(n_clusters=k, n_neighbors=30)})

    # E. BERTopic (Sweep across min_cluster_size)
    for mcs, nn in [(450, 30), (300, 30), (230, 12), (180, 12), (160, 11), (120, 11), (80, 10), (50, 10)]:
        configs.append({"family": "BERTopic", "variant": f"BERTopic (mcs={mcs})", "k_target": -1, "model": BERTopicStrategyClusterer(min_cluster_size=mcs, n_neighbors=nn)})

    log(f"Assembled {len(configs)} candidate configurations for systematic benchmark.")

    results_table = []

    # 4. Benchmark Loop
    for idx, cfg in enumerate(configs):
        t0 = time.time()
        family = cfg["family"]
        variant = cfg["variant"]
        k_target = cfg["k_target"]
        clusterer = cfg["model"]

        log(f"\n[{idx+1}/{len(configs)}] Running {variant} (Target K: {k_target})...")

        try:
            strat_ids = clusterer.fit_predict(all_trans_inputs)
            inferred_K = len(set(strat_ids))

            # Map window strategy IDs to continuous bouts
            bout_strat_ids = np.zeros(len(all_bouts_list), dtype=int)
            bout_idx = 0
            for s_idx, s in enumerate(raw_sessions):
                s_bouts = session_bouts[s_idx]
                s_mask = np.array([w["session_id"] == s.session_id for w in win_metadata])
                if not np.any(s_mask):
                    bout_idx += len(s_bouts)
                    continue
                s_win_mids = win_mids[s_mask]
                s_strat_ids = strat_ids[s_mask]
                for b in s_bouts:
                    bm = b["mid_frame"]
                    closest_w = np.argmin(np.abs(s_win_mids - bm))
                    bout_strat_ids[bout_idx] = s_strat_ids[closest_w]
                    bout_idx += 1

            # Autoregressive Forecasting
            s_train, s_test = bout_strat_ids[:n_train_bouts], bout_strat_ids[n_train_bouts:]
            ar_model = StrategyConditionedPoseModel(feature_dim=30, alpha=1.0)
            ar_model.fit(X_train_bouts, s_train)
            ar_metrics = ar_model.evaluate(X_test_bouts, s_test)

            # Zero-Knowledge Linear Probing
            K_dim = max(inferred_K, np.max(strat_ids) + 1)
            X_strat_onehot = np.zeros((len(strat_ids), K_dim), dtype=np.float32)
            for i, s_id in enumerate(strat_ids):
                X_strat_onehot[i, s_id] = 1.0
            probe_results = run_scalar_probing_battery(X_strat_onehot, all_raw_sylls, scalar_targets)

            # Session-Level Cohort Proportions
            props_full = np.zeros((len(session_ids_ordered), K_dim))
            props_post = np.zeros((len(session_ids_ordered), K_dim))
            for s_i, sid in enumerate(session_ids_ordered):
                s_w = strat_ids[session_groups[sid]["window_indices"]]
                if len(s_w) > 0:
                    c = pd.Series(s_w).value_counts(normalize=True)
                    for k_id, p in c.items():
                        props_full[s_i, k_id] = p
                s_post = strat_ids[session_groups[sid]["post_indices"]]
                if len(s_post) > 0:
                    c_post = pd.Series(s_post).value_counts(normalize=True)
                    for k_id, p in c_post.items():
                        props_post[s_i, k_id] = p

            acc_gt = eval_rf_classifier(props_full, labels_gt)
            acc_drug_post = eval_rf_classifier(props_post, labels_drug)

            elapsed = time.time() - t0
            r2_spatial = probe_results["Spatial Exploration Entropy"]["r2_strategy"]
            r2_cadence = probe_results["Switch Cadence"]["r2_strategy"]
            r2_chirality = probe_results["Turning Chirality"]["r2_strategy"]
            r2_seq_entropy = probe_results["Sequence Transition Entropy"]["r2_strategy"]
            info_gain = ar_metrics["relative_gain"] * 100.0

            log(f"  Inferred K: {inferred_K} | Info Gain: {info_gain:+.2f}% | R² Spatial: {r2_spatial:.4f} | Genotype Acc: {acc_gt*100:.1f}% | Drug Acc: {acc_drug_post*100:.1f}% ({elapsed:.1f}s)")

            res_entry = {
                "family": family,
                "variant": variant,
                "target_K": k_target,
                "inferred_K": inferred_K,
                "info_gain_pct": info_gain,
                "mse_unconditioned": ar_metrics["mse_unconditioned"],
                "mse_conditioned": ar_metrics["mse_conditioned"],
                "mse_shuffled": ar_metrics["mse_shuffled"],
                "cosine_similarity": ar_metrics["cosine_sim_conditioned"],
                "r2_spatial_entropy": r2_spatial,
                "r2_cadence": r2_cadence,
                "r2_chirality": r2_chirality,
                "r2_sequence_entropy": r2_seq_entropy,
                "acc_genotype": acc_gt,
                "acc_drug_post": acc_drug_post,
                "elapsed_sec": elapsed
            }
            results_table.append(res_entry)

            # Log step to W&B
            wandb.log({
                "family": family,
                "variant": variant,
                "K": inferred_K,
                "info_gain_pct": info_gain,
                "r2_spatial_entropy": r2_spatial,
                "r2_cadence": r2_cadence,
                "r2_sequence_entropy": r2_seq_entropy,
                "acc_genotype": acc_gt,
                "acc_drug_post": acc_drug_post,
                "cosine_similarity": ar_metrics["cosine_sim_conditioned"]
            })

        except Exception as e:
            log(f"  [ERROR] Trial failed: {e}")

    # 5. Export Results & Generate Plots
    df_results = pd.DataFrame(results_table)
    csv_path = os.path.join(out_dir, "multi_clustering_benchmark_results.csv")
    df_results.to_csv(csv_path, index=False)
    log(f"\nSaved benchmark results to: {csv_path}")

    # Multi-panel Resolution Comparison Curves
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    family_colors = {
        "GMM": "#ef4444",
        "KMeans": "#3b82f6",
        "Spectral": "#10b981",
        "Agglomerative": "#8b5cf6",
        "BERTopic": "#f59e0b"
    }

    # Panel 1: K vs. Spatial Exploration Entropy (R²)
    for fam in df_results["family"].unique():
        sub = df_results[df_results["family"] == fam].sort_values("inferred_K")
        axes[0, 0].plot(sub["inferred_K"], sub["r2_spatial_entropy"], marker="o", label=fam, color=family_colors.get(fam, "gray"), linewidth=2)
    axes[0, 0].axhline(0.6635, color="black", linestyle="--", label="Raw Syllables (100D)")
    axes[0, 0].set_title("Spatial Exploration Entropy (R²) vs. Resolution (K)", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("Inferred Number of Clusters (K)")
    axes[0, 0].set_ylabel("Cross-Validated R²")
    axes[0, 0].legend()
    axes[0, 0].grid(True, linestyle="--", alpha=0.5)

    # Panel 2: K vs. Autoregressive Information Gain (%)
    for fam in df_results["family"].unique():
        sub = df_results[df_results["family"] == fam].sort_values("inferred_K")
        axes[0, 1].plot(sub["inferred_K"], sub["info_gain_pct"], marker="s", label=fam, color=family_colors.get(fam, "gray"), linewidth=2)
    axes[0, 1].set_title("Relative Information Gain (%) in Next-Bout Forecasting", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("Inferred Number of Clusters (K)")
    axes[0, 1].set_ylabel("Gain vs. Unconditioned (%)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, linestyle="--", alpha=0.5)

    # Panel 3: K vs. Genotype Decodability Accuracy
    for fam in df_results["family"].unique():
        sub = df_results[df_results["family"] == fam].sort_values("inferred_K")
        axes[1, 0].plot(sub["inferred_K"], sub["acc_genotype"] * 100.0, marker="^", label=fam, color=family_colors.get(fam, "gray"), linewidth=2)
    axes[1, 0].axhline(50.0, color="gray", linestyle=":", label="Chance (50%)")
    axes[1, 0].set_title("Genotype Decodability (WT vs. DAT-KI)", fontsize=12, fontweight="bold")
    axes[1, 0].set_xlabel("Inferred Number of Clusters (K)")
    axes[1, 0].set_ylabel("Cross-Validated Accuracy (%)")
    axes[1, 0].legend()
    axes[1, 0].grid(True, linestyle="--", alpha=0.5)

    # Panel 4: K vs. Post-Injection Drug Decodability Accuracy
    for fam in df_results["family"].unique():
        sub = df_results[df_results["family"] == fam].sort_values("inferred_K")
        axes[1, 1].plot(sub["inferred_K"], sub["acc_drug_post"] * 100.0, marker="d", label=fam, color=family_colors.get(fam, "gray"), linewidth=2)
    axes[1, 1].axhline(50.0, color="gray", linestyle=":", label="Chance (50%)")
    axes[1, 1].set_title("Post-Injection Drug Decodability (Saline vs. Amph)", fontsize=12, fontweight="bold")
    axes[1, 1].set_xlabel("Inferred Number of Clusters (K)")
    axes[1, 1].set_ylabel("Cross-Validated Accuracy (%)")
    axes[1, 1].legend()
    axes[1, 1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plot_path = os.path.join(out_dir, "multi_clustering_resolution_curves.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    log(f"Saved diagnostic resolution curves to: {plot_path}")

    # Log figure and table to W&B
    wandb.log({
        "multi_clustering_resolution_curves": wandb.Image(plot_path),
        "benchmark_summary_table": wandb.Table(dataframe=df_results)
    })
    wandb.finish()

    elapsed_total = time.time() - start_time
    summary_msg = f"SUCCESS: Multi-Clustering Benchmark finished in {elapsed_total/60:.1f} min across {len(df_results)} trials. W&B logged!"
    log(summary_msg)
    send_ntfy(summary_msg, "Benchmark Complete")

if __name__ == "__main__":
    main()
