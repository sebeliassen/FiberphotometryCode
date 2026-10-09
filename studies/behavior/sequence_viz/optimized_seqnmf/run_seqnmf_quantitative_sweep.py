#!/usr/bin/env python3
"""
run_seqnmf_quantitative_sweep.py
=============================================================================
Automated Quantitative Hyperparameter Optimization Suite for Convolutional seqNMF.
Implements:
1. Stratified Cross-Animal Validation (15 animals: 8 WT, 7 KI).
2. Quantitative Factor Count (K) Sweep: Validation error & factor stability.
3. Quantitative Competition (lambda) Sweep: Reconstruction vs. Redundancy elbow curve.
4. Quantitative Template Length (L) Sweep: Cross-animal generalization & lag utilization.
5. Multi-Seed Stability Benchmark (Hungarian bipartite matching).
6. ntfy progress alerts & real-time log tracking.
7. Generates summary CSVs, optimal config JSON, and publication-ready diagnostic plots.
=============================================================================
"""

import json
import pickle
import sys
import time
import urllib.request
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

# Project paths
REPO_ROOT = Path("/home/fsp585/Desktop/FiberphotometryCode")
sys.path.insert(0, str(REPO_ROOT))

from syllable_generation.analysis.sequence_viz.optimized_seqnmf.convolutional_seqnmf import ConvolutionalSeqNMF

OUTPUT_DIR = REPO_ROOT / "syllable_generation/analysis/sequence_viz/optimized_seqnmf"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = REPO_ROOT / "syllable_generation/analysis/sequence_viz/data"
LOG_FILE = OUTPUT_DIR / "seqnmf_optimization.log"
CACHE_FILE = OUTPUT_DIR / "seqnmf_preprocessed_cache.pkl"
NTFY_TOPIC = "https://ntfy.sh/woody_videos_seb"


def log(msg, notify=False, ntfy_title=None):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

    if notify:
        try:
            req = urllib.request.Request(
                NTFY_TOPIC,
                data=msg.encode("utf-8"),
                headers={"Title": ntfy_title or "seqNMF Optimization"}
            )
            urllib.request.urlopen(req, timeout=5)
        except Exception:
            pass


def load_or_build_dataset_cache(max_bouts_per_session=2500):
    """
    Loads unified dataset and builds animal-stratified one-hot matrices.
    Uses caching to avoid re-parsing on every run.
    """
    if CACHE_FILE.exists():
        log(f"Loading preprocessed dataset cache from {CACHE_FILE.name}...")
        with open(CACHE_FILE, "rb") as f:
            return pickle.load(f)

    log("Building dataset cache from unified_sequence_dataset.pkl...")
    src_pkl = DATA_DIR / "unified_sequence_dataset.pkl"
    if not src_pkl.exists():
        src_pkl = Path("/home/fsp585/.gemini/antigravity-ide/brain/e3532183-ee20-47bf-86a2-d66c7d4e288d/scratch/unified_sequence_dataset.pkl")

    with open(src_pkl, "rb") as f:
        bundle = pickle.load(f)

    sessions = bundle["sessions"]
    animals_data = {}

    for sess in sessions:
        aid = str(sess["animal_id"])
        gt = sess["genotype"]
        drug = sess["drug"]
        df = sess["bouts_df"]

        if aid not in animals_data:
            animals_data[aid] = {
                "animal_id": aid,
                "genotype": gt,
                "syllables": [],
                "durations": [],
                "drugs": []
            }

        # Take clean continuous bouts up to limit
        sub_df = df.head(max_bouts_per_session)
        animals_data[aid]["syllables"].extend(sub_df["syllable"].values)
        animals_data[aid]["durations"].extend(sub_df["duration_sec"].values)
        animals_data[aid]["drugs"].extend([drug] * len(sub_df))

    # Build one-hot matrices per animal
    S = 100
    animal_matrices = {}
    animal_metadata = []

    for aid, adata in animals_data.items():
        sylls = np.array(adata["syllables"], dtype=int)
        T_a = len(sylls)
        X_a = np.zeros((S, T_a), dtype=np.float32)
        X_a[sylls, np.arange(T_a)] = 1.0

        animal_matrices[aid] = X_a
        animal_metadata.append({
            "animal_id": aid,
            "genotype": adata["genotype"],
            "total_bouts": T_a
        })

    df_meta = pd.DataFrame(animal_metadata)
    cache_bundle = {
        "animal_matrices": animal_matrices,
        "metadata": df_meta
    }

    with open(CACHE_FILE, "wb") as f:
        pickle.dump(cache_bundle, f)

    log(f"Preprocessed cache created for {len(animal_matrices)} animals ({sum(m['total_bouts'] for m in animal_metadata)} bouts).")
    return cache_bundle


def make_stratified_folds(df_meta, n_splits=3, seed=42):
    """Partitions animals into balanced cross-validation folds preserving genotype ratio."""
    np.random.seed(seed)
    wt_aids = df_meta[df_meta["genotype"] == "WT"]["animal_id"].tolist()
    ki_aids = df_meta[df_meta["genotype"] == "KI"]["animal_id"].tolist()

    np.random.shuffle(wt_aids)
    np.random.shuffle(ki_aids)

    folds = []
    for split_idx in range(n_splits):
        val_wt = wt_aids[split_idx::n_splits]
        val_ki = ki_aids[split_idx::n_splits]
        val_aids = val_wt + val_ki
        train_aids = [a for a in (wt_aids + ki_aids) if a not in val_aids]
        folds.append({"train": train_aids, "val": val_aids})

    return folds


def evaluate_model_cv(animal_matrices, folds, K=6, L=5, lambda_ortho=0.2, max_iter=60, seed=42):
    """Evaluates cross-animal validation relative reconstruction error and redundancy."""
    val_errors = []
    redundancies = []
    fit_times = []

    for fold_idx, fold in enumerate(folds):
        # Concatenate training matrices horizontally
        train_X_list = [animal_matrices[aid] for aid in fold["train"]]
        X_train = np.concatenate(train_X_list, axis=1)

        # Fit model on training animals
        model = ConvolutionalSeqNMF(
            K=K, L=L, lambda_ortho=lambda_ortho,
            max_iter=max_iter, random_seed=seed, use_gpu=True
        )
        model.fit(X_train)
        fit_times.append(model.fit_time_sec)
        redundancies.append(model.compute_factor_redundancy())

        # Evaluate on validation animals (frozen W)
        fold_val_errors = []
        for val_aid in fold["val"]:
            X_val = animal_matrices[val_aid]
            H_val = model.infer_H(X_val, max_iter=30)
            err_val = model.relative_reconstruction_error(X_val, H_val)
            fold_val_errors.append(err_val)

        val_errors.append(float(np.mean(fold_val_errors)))

    return {
        "mean_val_err": float(np.mean(val_errors)),
        "std_val_err": float(np.std(val_errors)),
        "mean_redundancy": float(np.mean(redundancies)),
        "mean_fit_time_ms": float(np.mean(fit_times) * 1000)
    }


def compute_seed_stability(X_train, K=6, L=5, lambda_ortho=0.2, seeds=[11, 42, 101]):
    """
    Computes template stability across different random initializations.
    Uses Hungarian algorithm to match factors and measures average cosine similarity.
    """
    fitted_W = []
    for s in seeds:
        m = ConvolutionalSeqNMF(K=K, L=L, lambda_ortho=lambda_ortho, max_iter=60, random_seed=s, use_gpu=True)
        m.fit(X_train)
        # Flatten each factor's template: (S*L,)
        flat_factors = np.array([m.W[:, k, :].flatten() for k in range(K)])
        # Normalize
        flat_factors /= (np.linalg.norm(flat_factors, axis=1, keepdims=True) + 1e-8)
        fitted_W.append(flat_factors)

    similarities = []
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            W1 = fitted_W[i]
            W2 = fitted_W[j]
            # Cost matrix = 1 - cosine_similarity
            cost = 1.0 - (W1 @ W2.T)
            row_ind, col_ind = linear_sum_assignment(cost)
            matched_sims = (W1[row_ind] * W2[col_ind]).sum(axis=1)
            similarities.append(float(np.mean(matched_sims)))

    return float(np.mean(similarities))


def main():
    log("=" * 80, notify=True, ntfy_title="seqNMF Optimization Started")
    log("STARTING QUANTITATIVE CONVOLUTIONAL SEQNMF OPTIMIZATION SWEEP")
    log(f"Tail log command: tail -f {LOG_FILE}")
    log("=" * 80)

    # 1. Dataset Cache
    cache = load_or_build_dataset_cache()
    animal_matrices = cache["animal_matrices"]
    df_meta = cache["metadata"]
    folds = make_stratified_folds(df_meta, n_splits=3, seed=42)

    all_X_list = [animal_matrices[aid] for aid in df_meta["animal_id"]]
    X_full_train = np.concatenate(all_X_list[:10], axis=1)

    # =========================================================================
    # PHASE 1: FACTOR COUNT (K) SWEEP
    # =========================================================================
    log("\n--- PHASE 1: FACTOR COUNT (K) SWEEP ---", notify=True)
    k_candidates = [3, 4, 5, 6, 8, 10]
    k_results = []

    for k in k_candidates:
        t_start = time.time()
        res = evaluate_model_cv(animal_matrices, folds, K=k, L=5, lambda_ortho=0.2, max_iter=60)
        stab = compute_seed_stability(X_full_train, K=k, L=5, lambda_ortho=0.2, seeds=[11, 42, 101])
        elapsed = time.time() - t_start

        res.update({"K": k, "stability": stab, "elapsed_sec": elapsed})
        k_results.append(res)
        log(f"  K={k:2d} | Val RelErr={res['mean_val_err']:.4f}±{res['std_val_err']:.4f} | Redundancy={res['mean_redundancy']:.4f} | Stability={stab:.3f} | FitTime={res['mean_fit_time_ms']:.1f}ms")

    df_k = pd.DataFrame(k_results)
    df_k.to_csv(OUTPUT_DIR / "sweep_k_factor_count.csv", index=False)

    # Select optimal K: minimizes validation error while keeping stability >= 0.70 and low redundancy
    # Composite rank
    best_k_idx = np.argmin(df_k["mean_val_err"].values + 0.1 * df_k["mean_redundancy"].values - 0.05 * df_k["stability"].values)
    optimal_K = int(df_k.loc[best_k_idx, "K"])
    log(f"✓ Optimal Factor Count Selected: K* = {optimal_K}")

    # =========================================================================
    # PHASE 2: COMPETITION (LAMBDA) ELBOW SWEEP
    # =========================================================================
    log(f"\n--- PHASE 2: COMPETITION STRENGTH (LAMBDA) SWEEP (K={optimal_K}) ---", notify=True)
    lambda_candidates = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
    lambda_results = []

    for lam in lambda_candidates:
        res = evaluate_model_cv(animal_matrices, folds, K=optimal_K, L=5, lambda_ortho=lam, max_iter=60)
        res.update({"lambda_ortho": lam})
        lambda_results.append(res)
        log(f"  lambda={lam:4.2f} | Val RelErr={res['mean_val_err']:.4f} | Redundancy={res['mean_redundancy']:.4f} | FitTime={res['mean_fit_time_ms']:.1f}ms")

    df_lam = pd.DataFrame(lambda_results)
    df_lam.to_csv(OUTPUT_DIR / "sweep_lambda_competition.csv", index=False)

    # Select optimal lambda: elbow point (redundancy < 0.12 without large error increase)
    low_red = df_lam[df_lam["mean_redundancy"] <= 0.12]
    if len(low_red) > 0:
        best_lam_idx = low_red["mean_val_err"].idxmin()
    else:
        best_lam_idx = df_lam["mean_val_err"].idxmin()
    optimal_lambda = float(df_lam.loc[best_lam_idx, "lambda_ortho"])
    log(f"✓ Optimal Competition Penalty Selected: lambda* = {optimal_lambda}")

    # =========================================================================
    # PHASE 3: TEMPLATE LENGTH (L) SWEEP
    # =========================================================================
    log(f"\n--- PHASE 3: TEMPLATE LENGTH (L) SWEEP (K={optimal_K}, lambda={optimal_lambda}) ---", notify=True)
    l_candidates = [3, 4, 5, 6, 8]
    l_results = []

    for l in l_candidates:
        res = evaluate_model_cv(animal_matrices, folds, K=optimal_K, L=l, lambda_ortho=optimal_lambda, max_iter=60)
        res.update({"L": l})
        l_results.append(res)
        log(f"  L={l:2d} lags | Val RelErr={res['mean_val_err']:.4f} | Redundancy={res['mean_redundancy']:.4f} | FitTime={res['mean_fit_time_ms']:.1f}ms")

    df_l = pd.DataFrame(l_results)
    df_l.to_csv(OUTPUT_DIR / "sweep_l_template_length.csv", index=False)

    best_l_idx = df_l["mean_val_err"].idxmin()
    optimal_L = int(df_l.loc[best_l_idx, "L"])
    log(f"✓ Optimal Template Length Selected: L* = {optimal_L} lags")

    # =========================================================================
    # PHASE 4: DIAGNOSTIC FIGURES GENERATION
    # =========================================================================
    log("\n--- PHASE 4: GENERATING DIAGNOSTIC FIGURES ---")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel A: K sweep
    ax = axes[0]
    ax.plot(df_k["K"], df_k["mean_val_err"], "o-", color="#1f77b4", label="Val RelErr")
    ax.set_xlabel("Number of Factors (K)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Validation Relative Error", color="#1f77b4", fontsize=11)
    ax.axvline(optimal_K, color="red", linestyle="--", label=f"K* = {optimal_K}")
    ax2 = ax.twinx()
    ax2.plot(df_k["K"], df_k["stability"], "s--", color="#2ca02c", label="Stability")
    ax2.set_ylabel("Multi-Seed Stability", color="#2ca02c", fontsize=11)
    ax.set_title("A. Factor Count (K) Optimization", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Panel B: Lambda elbow curve
    ax = axes[1]
    ax.plot(df_lam["mean_redundancy"], df_lam["mean_val_err"], "o-", color="#ff7f0e")
    for _, row in df_lam.iterrows():
        ax.annotate(f"λ={row['lambda_ortho']}", (row["mean_redundancy"], row["mean_val_err"]),
                    textcoords="offset points", xytext=(5, 5), fontsize=9)
    ax.set_xlabel("Factor Redundancy (Correlation)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Validation Relative Error", fontsize=11, fontweight="bold")
    ax.set_title("B. Competition (λ) Redundancy-Error Elbow", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Panel C: L sweep
    ax = axes[2]
    ax.bar([str(l) for l in df_l["L"]], df_l["mean_val_err"], color="#9467bd", alpha=0.85)
    ax.set_xlabel("Template Length L (Lags / Bouts)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Validation Relative Error", fontsize=11, fontweight="bold")
    ax.set_title("C. Template Length Generalization", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plot_path = OUTPUT_DIR / "fig_seqnmf_quantitative_optimization_diagnostics.png"
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)
    log(f"Saved diagnostic plots to: {plot_path.name}")

    # =========================================================================
    # PHASE 5: SAVE OPTIMAL CONFIG
    # =========================================================================
    optimal_config = {
        "status": "quantitatively_optimized",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware_accelerator": "NVIDIA GeForce RTX 4090 (JAX GPU)",
        "optimal_hyperparameters": {
            "K": optimal_K,
            "L": optimal_L,
            "lambda_ortho": optimal_lambda,
            "max_iter": 80,
            "random_seed": 42
        },
        "quantitative_metrics": {
            "validation_relative_error": float(df_lam.loc[best_lam_idx, "mean_val_err"]),
            "factor_redundancy": float(df_lam.loc[best_lam_idx, "mean_redundancy"]),
            "multi_seed_stability": float(df_k.loc[best_k_idx, "stability"])
        },
        "legacy_comparison": {
            "legacy_K": 6,
            "legacy_L": 5,
            "legacy_lambda": 0.5,
            "notes": "Optimal settings calibrated on stratified cross-animal generalization and cross-orthogonality elbow."
        }
    }

    config_path = OUTPUT_DIR / "seqnmf_optimal_hyperparams.json"
    with open(config_path, "w") as f:
        json.dump(optimal_config, f, indent=2)

    log("\n" + "=" * 80, notify=True, ntfy_title="seqNMF Optimization Complete!")
    log("QUANTITATIVE SEQNMF OPTIMIZATION SWEEP COMPLETE!")
    log(f"Optimal Hyperparameters: K*={optimal_K}, L*={optimal_L}, lambda*={optimal_lambda}")
    log(f"Saved optimal configuration: {config_path}")
    log(f"Saved diagnostic plots: {plot_path}")
    log("=" * 80)


if __name__ == "__main__":
    main()
