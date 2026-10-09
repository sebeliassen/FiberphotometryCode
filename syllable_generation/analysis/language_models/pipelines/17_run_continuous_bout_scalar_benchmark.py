import os
import pickle
import sys
import time

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import requests

ROOT_DIR = "/home/fsp585/Desktop/FiberphotometryCode"
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "syllable_generation/analysis/language_models"))

from lib.evaluation.scalar_probes import compute_window_scalars, run_scalar_probing_battery
from lib.models.continuous import StrategyConditionedPoseModel
from lib.models.strategies import StrategyBERTopic
from lib.preprocessing.bout_features import build_and_cache_dataset_bouts

# cuML / GPU Acceleration imports
try:
    from cuml.cluster import HDBSCAN as GPU_HDBSCAN
    from cuml.manifold import UMAP as GPU_UMAP
    HAS_GPU = True
except ImportError:
    HAS_GPU = False

from fpcode.common.notify import send_ntfy

def main():
    start_time = time.time()
    out_dir = os.path.join(ROOT_DIR, "syllable_generation/analysis/language_models/output/strategies_extended")
    os.makedirs(out_dir, exist_ok=True)
    log_file = os.path.join(out_dir, "logs/continuous_scalar_benchmark.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    def log(msg: str):
        ts = time.strftime('%Y-%m-%d %H:%M:%S')
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        with open(log_file, "a") as f:
            f.write(line + "\n")

    log("="*70)
    log("Starting Continuous Bout & Zero-Knowledge Scalar Probing Benchmark")
    log(f"GPU Acceleration (cuML): {HAS_GPU}")
    send_ntfy("Starting Continuous Bout & Zero-Knowledge Scalar Probing Benchmark", "Pipeline Started")

    # 1. Load Sessions
    pkl_path = os.path.join(ROOT_DIR, "syllable_generation/data/jonathan_amph_sessions_v2_w_sylls.pkl")
    if not os.path.exists(pkl_path):
        pkl_path = os.path.join(ROOT_DIR, "syllable_generation/data/jonathan_amph_sessions_w_sylls.pkl")
    log(f"Loading session data from: {pkl_path}")
    with open(pkl_path, "rb") as f:
        sessions = pickle.load(f)
    log(f"Loaded {len(sessions)} sessions.")

    # 2. Extract & Cache 30-D Continuous Bout Sequences
    cache_path = os.path.join(out_dir, "continuous_bout_cache.pkl")
    session_bouts, b2v = build_and_cache_dataset_bouts(sessions, cache_path=cache_path, force_recompute=False)

    # 3. Extract 128s Sliding Windows & Ground Truth Behavioral Scalars
    log("Building 128s sliding windows and extracting zero-knowledge behavioral scalars...")
    window_scalars_cache = os.path.join(out_dir, "window_scalars_cache.pkl")

    if os.path.exists(window_scalars_cache):
        with open(window_scalars_cache, "rb") as f:
            scalar_cache_data = pickle.load(f)
            all_win_scalars = scalar_cache_data["scalars"]
            all_raw_sylls = scalar_cache_data["raw_sylls"]
            all_trans_inputs = scalar_cache_data["trans_inputs"]
            win_metadata = scalar_cache_data["meta"]
        log(f"[CACHE HIT] Loaded {len(all_win_scalars)} window scalars from cache.")
    else:
        all_win_scalars = []
        all_raw_sylls = []
        all_trans_inputs = []
        win_metadata = []

        WINDOW_FRAMES = 10240  # 128s at 80.025 Hz
        STRIDE_FRAMES = 800    # 10s stride

        for s_idx, s in enumerate(sessions):
            df = s.all_data_df
            syll_df = s.syllables
            first_valid = df['centroid_x'].first_valid_index()
            last_valid = df['centroid_x'].last_valid_index()
            if first_valid is None or last_valid is None:
                continue

            n_frames = last_valid - first_valid + 1
            if n_frames < WINDOW_FRAMES:
                continue

            # Syllable transitions & raw counts
            for sf in range(first_valid, last_valid - WINDOW_FRAMES + 1, STRIDE_FRAMES):
                ef = sf + WINDOW_FRAMES
                # 4 Scalars
                scalars = compute_window_scalars(s, sf, ef)
                all_win_scalars.append(scalars)

                # Raw Syllable Frequencies (100-D)
                sylls = syll_df.loc[sf:ef, 'syllable'].values
                valid_sylls = sylls[~pd.isna(sylls)].astype(int)
                raw_counts = np.bincount(valid_sylls, minlength=100)[:100].astype(np.float32)
                raw_freq = raw_counts / (np.sum(raw_counts) + 1e-12)
                all_raw_sylls.append(raw_freq)

                # 1-step Transition matrix (10,000-D flattened)
                trans_mat = np.zeros((100, 100), dtype=np.float32)
                if len(valid_sylls) > 1:
                    for i in range(len(valid_sylls) - 1):
                        u, v = valid_sylls[i], valid_sylls[i+1]
                        if u < 100 and v < 100:
                            trans_mat[u, v] += 1.0
                    r_sums = trans_mat.sum(axis=1, keepdims=True)
                    trans_mat = np.divide(trans_mat, r_sums, out=np.zeros_like(trans_mat), where=r_sums > 0)
                all_trans_inputs.append(trans_mat.flatten())

                win_metadata.append({
                    "session_idx": s_idx,
                    "session_id": s.session_id,
                    "start_frame": sf,
                    "end_frame": ef,
                    "mid_frame": (sf + ef) // 2
                })

        all_raw_sylls = np.array(all_raw_sylls)
        all_trans_inputs = np.array(all_trans_inputs)
        with open(window_scalars_cache, "wb") as f:
            pickle.dump({
                "scalars": all_win_scalars,
                "raw_sylls": all_raw_sylls,
                "trans_inputs": all_trans_inputs,
                "meta": win_metadata
            }, f)
        log(f"[CACHED] Saved {len(all_win_scalars)} windows to {window_scalars_cache}")

    # Build target arrays for the 4 scalars
    scalar_targets = {
        "Spatial Exploration Entropy": np.array([s["spatial_entropy"] for s in all_win_scalars]),
        "Switch Cadence": np.array([s["cadence"] for s in all_win_scalars]),
        "Turning Chirality": np.array([s["chirality"] for s in all_win_scalars]),
        "Sequence Transition Entropy": np.array([s["sequence_entropy"] for s in all_win_scalars])
    }

    # 4. Fit Models across Granularities: K ≈ 3, 9, 20 (Testing min_cluster_size sweep)
    candidate_configs = [
        {"name": "3-Vibe Model", "mcs": 450, "nn": 30, "nc": 5},
        {"name": "9-Vibe Model", "mcs": 230, "nn": 12, "nc": 10},
        {"name": "20-Vibe Model", "mcs": 160, "nn": 11, "nc": 10}
    ]

    benchmark_rows = []

    # Flatten all bouts for autoregressive forecasting
    all_bouts_list = [b for s_bouts in session_bouts for b in s_bouts]
    all_bout_vectors = np.array([b["vector"] for b in all_bouts_list])
    log(f"Total flattened continuous bouts: {len(all_bout_vectors)} (Dim: {all_bout_vectors.shape[1]})")

    for cfg in candidate_configs:
        log(f"\n--- Evaluating Candidate: {cfg['name']} (mcs={cfg['mcs']}, nn={cfg['nn']}, nc={cfg['nc']}) ---")

        # Fit BERTopic on Transition Inputs
        model = StrategyBERTopic(
            n_neighbors=cfg["nn"],
            min_cluster_size=cfg["mcs"],
            min_samples=15,
            n_components=cfg["nc"],
            random_state=42
        )
        strat_ids = model.fit_predict(all_trans_inputs)
        inferred_K = len(set(strat_ids))
        log(f"Inferred Active Clusters (K): {inferred_K}")

        # Assign strategy IDs to each continuous bout based on frame temporal overlap
        log("Mapping window strategy assignments to continuous bouts...")
        bout_strat_ids = np.zeros(len(all_bouts_list), dtype=int)

        # Window midpoint spatial lookup
        win_mids = np.array([w["mid_frame"] for w in win_metadata])
        win_sids = [w["session_id"] for w in win_metadata]

        # Fast session-wise mapping
        bout_idx = 0
        for s_idx, s in enumerate(sessions):
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

        # 5. Evaluate Continuous Autoregressive Forecasting
        log("Fitting Strategy-Conditioned Autoregressive Pose Model...")
        # 80/20 train/test split on bouts
        n_train = int(0.8 * len(all_bout_vectors))
        X_train_bouts, X_test_bouts = all_bout_vectors[:n_train], all_bout_vectors[n_train:]
        s_train, s_test = bout_strat_ids[:n_train], bout_strat_ids[n_train:]

        ar_model = StrategyConditionedPoseModel(feature_dim=30, alpha=1.0)
        ar_model.fit(X_train_bouts, s_train)
        ar_metrics = ar_model.evaluate(X_test_bouts, s_test)

        log(f"  Autoregressive MSE (Unconditioned): {ar_metrics['mse_unconditioned']:.5f}")
        log(f"  Autoregressive MSE (Strategy-Conditioned): {ar_metrics['mse_conditioned']:.5f}")
        log(f"  Autoregressive MSE (Shuffled Control): {ar_metrics['mse_shuffled']:.5f}")
        log(f"  Relative Information Gain: {ar_metrics['relative_gain']*100:.2f}%")
        log(f"  Next-Bout Cosine Similarity: {ar_metrics['cosine_sim_conditioned']:.4f}")

        # 6. Evaluate Zero-Knowledge Linear Probes
        log("Running 4-Scalar Linear Probing Battery...")
        # Build one-hot window strategy representations
        K_dim = max(inferred_K, np.max(strat_ids) + 1)
        X_strat_onehot = np.zeros((len(strat_ids), K_dim), dtype=np.float32)
        for i, s_id in enumerate(strat_ids):
            X_strat_onehot[i, s_id] = 1.0

        probe_results = run_scalar_probing_battery(X_strat_onehot, all_raw_sylls, scalar_targets)

        for scalar_name, res in probe_results.items():
            log(f"  Scalar: {scalar_name}")
            log(f"    R² (Null Control):    {res['r2_null']:.4f}")
            log(f"    R² (Raw Syllables):   {res['r2_raw_syllables']:.4f}")
            log(f"    R² (Strategy Model):  {res['r2_strategy']:.4f} (Advantage: {res['compression_advantage']:+.4f})")

            benchmark_rows.append({
                "model_name": cfg["name"],
                "inferred_K": inferred_K,
                "scalar": scalar_name,
                "r2_null": res["r2_null"],
                "r2_raw_syllables": res["r2_raw_syllables"],
                "r2_strategy": res["r2_strategy"],
                "compression_advantage": res["compression_advantage"],
                "relative_info_gain_pct": ar_metrics["relative_gain"] * 100.0,
                "cosine_similarity": ar_metrics["cosine_sim_conditioned"]
            })

    # 7. Save Summary Table & Diagnostic Visualizations
    results_df = pd.DataFrame(benchmark_rows)
    csv_path = os.path.join(out_dir, "continuous_scalar_benchmark_results.csv")
    results_df.to_csv(csv_path, index=False)
    log(f"\nSaved full benchmark results to: {csv_path}")

    # Plot Visualizations
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: R² across the 4 scalars comparing models
    pivot_df = results_df.pivot(index="scalar", columns="model_name", values="r2_strategy")
    raw_s = results_df.groupby("scalar")["r2_raw_syllables"].mean()
    pivot_df["Raw Syllables (100D)"] = raw_s

    pivot_df.plot(kind="bar", ax=axes[0], colormap="viridis", width=0.8)
    axes[0].set_title("Zero-Knowledge Scalar Decodability (R² Score)", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Cross-Validated R²")
    axes[0].set_xlabel("")
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)
    axes[0].tick_params(axis="x", rotation=25)

    # Right: Relative Information Gain & Next-Bout Cosine Similarity
    ar_summary = results_df[["model_name", "inferred_K", "relative_info_gain_pct", "cosine_similarity"]].drop_duplicates()
    x = np.arange(len(ar_summary))
    axes[1].bar(x - 0.15, ar_summary["relative_info_gain_pct"], width=0.3, label="Relative Info Gain (%)", color="#3b82f6")
    ax2 = axes[1].twinx()
    ax2.plot(x + 0.15, ar_summary["cosine_similarity"], marker="o", color="#ef4444", linewidth=2.5, label="Next-Bout Cosine Sim")

    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"{r['model_name']}\n(K={r['inferred_K']})" for _, r in ar_summary.iterrows()])
    axes[1].set_ylabel("Relative Information Gain (%)", color="#3b82f6", fontweight="bold")
    ax2.set_ylabel("Cosine Similarity", color="#ef4444", fontweight="bold")
    axes[1].set_title("Autoregressive Continuous Forecasting", fontsize=13, fontweight="bold")
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plot_path = os.path.join(out_dir, "scalar_probe_benchmark.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    log(f"Saved diagnostic figure to: {plot_path}")

    elapsed = time.time() - start_time
    summary_msg = f"SUCCESS: Continuous Bout & Scalar Benchmark complete in {elapsed/60:.1f} min! Results saved to {csv_path}"
    log(summary_msg)
    send_ntfy(summary_msg, "Benchmark Complete")

if __name__ == "__main__":
    main()
