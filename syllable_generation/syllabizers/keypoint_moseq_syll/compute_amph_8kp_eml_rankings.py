#!/usr/bin/env python3
"""
compute_amph_8kp_eml_rankings.py

Full end-to-end KPMS model evaluation, results extraction, and Expected Marginal Likelihood (EML) ranking
for the 8-Keypoint models in amph_project_8kp/:
1. Discovers runs (amph_8kp_k150k_run_01, run_02, run_03).
2. Verifies completion (max_iter >= 200).
3. Performs in-place HDF5 reindexing of syllables by frequency (runlength=True).
4. Extracts and saves `results.h5` for each completed model.
5. Computes pairwise cross-marginal log-likelihoods mll(i, j) on GPU.
6. Computes EML scores and standard errors.
7. Generates ranking CSV and visualization plots (eml_scores_8kp.png/pdf, heatmap).
8. Generates comprehensive markdown report.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.35"

import sys
import glob
import json
import time
import urllib.request
from pathlib import Path

# Fix numpy bool8 deprecation for modern numpy
import numpy as np
np.bool8 = np.bool_

import h5py
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import keypoint_moseq as kpms
from keypoint_moseq.fitting import marginal_log_likelihood
from keypoint_moseq.util import get_frequencies
from keypoint_moseq.io import load_hdf5, save_hdf5, _get_path
from jax_moseq.utils import unbatch

NTFY_URL = "https://ntfy.sh/woody_videos_seb"

def send_ntfy(title: str, msg: str, priority: str = "default", tags: str = "robot"):
    try:
        req = urllib.request.Request(
            NTFY_URL,
            data=msg.encode("utf-8"),
            headers={"Title": title, "Priority": priority, "Tags": tags}
        )
        urllib.request.urlopen(req, timeout=5)
    except Exception as e:
        print(f"[ntfy notice] (Quota/Network): {e}")

PROJECT_DIR = Path("/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/syllabizers/keypoint_moseq_syll/amph_project_8kp")
OUTPUT_DIR = PROJECT_DIR
PROGRESS_FILE = OUTPUT_DIR / "eml_pairwise_progress_8kp.json"

def fast_reindex_checkpoint(ckpt_path: Path):
    """
    High-performance in-place HDF5 reindexing.
    Modifies only betas, pi, Ab, Q, and z directly within HDF5 blocks.
    """
    t0 = time.time()
    with h5py.File(ckpt_path, "r+") as f:
        saved_iterations = [int(i) for i in f["model_snapshots"]]
        last_iter = np.max(saved_iterations)
        num_states = f[f"model_snapshots/{last_iter}/params/pi"].shape[0]
        z = f[f"model_snapshots/{last_iter}/states/z"][()]
        mask = f["data/mask"][()]
        
        freqs = get_frequencies(z, mask, num_states, runlength=True)
        index = np.argsort(freqs)[::-1]
        
        if np.all(index == np.arange(len(index))):
            return index, 0.0, True
            
        inv_index = np.argsort(index)
        for iteration in saved_iterations:
            snap = f[f"model_snapshots/{iteration}"]
            snap["params/betas"][:] = snap["params/betas"][()][index]
            snap["params/pi"][:] = snap["params/pi"][()][index, :][:, index]
            snap["params/Ab"][:] = snap["params/Ab"][()][index]
            snap["params/Q"][:] = snap["params/Q"][()][index]
            snap["states/z"][:] = inv_index[snap["states/z"][()]]
            
    return index, time.time() - t0, False

def fast_extract_results(ckpt_path: Path, results_path: Path):
    """
    High-performance results extraction.
    """
    t0 = time.time()
    if results_path.exists():
        results_path.unlink()
        
    with h5py.File(ckpt_path, "r") as f:
        last_iter = np.max([int(i) for i in f["model_snapshots"]])
        x = f[f"model_snapshots/{last_iter}/states/x"][()]
        z = f[f"model_snapshots/{last_iter}/states/z"][()]
        v = f[f"model_snapshots/{last_iter}/states/v"][()]
        h = f[f"model_snapshots/{last_iter}/states/h"][()]

    metadata = load_hdf5(str(ckpt_path), "metadata")

    nlags = x.shape[1] - z.shape[1]
    z_padded = np.pad(z, ((0, 0), (nlags, 0)), mode="edge")
    syllables = unbatch(z_padded, *metadata)
    latent_state = unbatch(x, *metadata)
    centroid = unbatch(v, *metadata)
    heading = unbatch(h, *metadata)

    results_dict = {
        rec: {
            "syllable": syllables[rec],
            "latent_state": latent_state[rec],
            "centroid": centroid[rec],
            "heading": heading[rec],
        }
        for rec in syllables.keys()
    }

    save_hdf5(str(results_path), results_dict, exist_ok=True)
    return time.time() - t0

def main():
    print("=" * 80)
    print("  8-KEYPOINT KPMS EXPECTED MARGINAL LIKELIHOOD (EML) MODEL SELECTION")
    print(f"  Project Directory: {PROJECT_DIR}")
    print(f"  Start Time:        {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # 1. Discover all candidate model directories
    all_model_dirs = sorted(glob.glob(str(PROJECT_DIR / "amph_8kp_k150k_run_*")))
    all_model_names = [os.path.basename(p) for p in all_model_dirs]
    print(f"\n[Audit] Found {len(all_model_names)} candidate model runs:")

    completed_models = []
    discarded_models = []

    for name in all_model_names:
        ckpt_path = PROJECT_DIR / name / "checkpoint.h5"
        summary_path = PROJECT_DIR / name / "model_summary.json"
        
        if not ckpt_path.exists():
            discarded_models.append((name, "No checkpoint.h5 found"))
            continue
            
        with h5py.File(ckpt_path, "r") as f:
            snaps = sorted([int(i) for i in f.get("model_snapshots", {}).keys()])
            max_iter = max(snaps) if snaps else 0
            
        seed = None
        if summary_path.exists():
            try:
                with open(summary_path) as sf:
                    seed = json.load(sf).get("seed")
            except Exception:
                pass

        if max_iter >= 200:
            completed_models.append((name, max_iter, seed))
            print(f"  [VALID]   {name}: max_iter={max_iter}, seed={seed}")
        else:
            discarded_models.append((name, f"Incomplete (max_iter={max_iter} < 200)"))
            print(f"  [DISCARD] {name}: max_iter={max_iter} < 200")

    print(f"\nSummary: {len(completed_models)} completed models retained for EML ranking; {len(discarded_models)} discarded.")

    if len(completed_models) < 2:
        print(f"[Error] Need at least 2 completed models for cross-EML, but found {len(completed_models)}. Exiting.")
        sys.exit(1)

    # 2. Reindex syllables and extract results for each completed model
    print("\n" + "=" * 80)
    print("  PHASE 1: SYLLABLE REINDEXING & RESULTS EXTRACTION")
    print("=" * 80)
    
    for name, max_iter, seed in completed_models:
        ckpt_path = PROJECT_DIR / name / "checkpoint.h5"
        res_path = PROJECT_DIR / name / "results.h5"
        
        idx, reindex_time, was_already = fast_reindex_checkpoint(ckpt_path)
        status_str = "Already reindexed" if was_already else f"Reindexed in {reindex_time:.3f}s"
        
        extract_time = fast_extract_results(ckpt_path, res_path)
        res_size_mb = res_path.stat().st_size / (1024 * 1024)
        print(f"  [{name}] {status_str} | Extracted results.h5 ({res_size_mb:.1f} MB) in {extract_time:.2f}s")

    # 3. Load lightweight model data for EML evaluation
    print("\n" + "=" * 80)
    print("  PHASE 2: PRE-STAGING DATA & PARAMETERS FOR GPU EML")
    print("=" * 80)
    
    first_ckpt = PROJECT_DIR / completed_models[0][0] / "checkpoint.h5"
    with h5py.File(first_ckpt, "r") as f:
        data_mask = f["data/mask"][()]
    mask_gpu = jnp.array(data_mask)
    print(f"  Loaded dataset mask: shape={data_mask.shape}, active frames={data_mask.sum():,}")

    model_xs = []
    model_params = []
    eval_model_names = [m[0] for m in completed_models]
    num_models = len(eval_model_names)

    t_load_start = time.time()
    for name, max_iter, seed in completed_models:
        ckpt_path = PROJECT_DIR / name / "checkpoint.h5"
        with h5py.File(ckpt_path, "r") as f:
            snap = f[f"model_snapshots/{max_iter}"]
            x = snap["states/x"][()]
            Ab = snap["params/Ab"][()]
            Q = snap["params/Q"][()]
            pi = snap["params/pi"][()]
            
        model_xs.append(x)
        model_params.append({
            "Ab": jnp.array(Ab),
            "Q": jnp.array(Q),
            "pi": jnp.array(pi)
        })
    print(f"  Loaded latents x (D={model_xs[0].shape[-1]}) and parameters for all {num_models} models in {time.time() - t_load_start:.2f}s.")

    # 4. Pairwise Marginal Log-Likelihood Computation
    print("\n" + "=" * 80)
    print(f"  PHASE 3: COMPUTING EXPECTED MARGINAL LIKELIHOOD MATRIX ({num_models}x{num_models} = {num_models * (num_models - 1)} pairs)")
    print("=" * 80)

    mlls = np.zeros((num_models, num_models), dtype=np.float64)
    completed_pairs = 0
    total_pairs = num_models * (num_models - 1)

    print("  Warming up JAX JIT compilation on GPU...")
    t_jit = time.time()
    _ = marginal_log_likelihood(
        mask_gpu[:10],
        jnp.array(model_xs[0][:10]),
        model_params[0]["Ab"],
        model_params[0]["Q"],
        model_params[0]["pi"]
    ).block_until_ready()
    print(f"  JIT warm-up complete in {time.time() - t_jit:.2f}s.")

    pair_times = []
    eml_start_time = time.time()

    for i in range(num_models):
        name_i = eval_model_names[i]
        params_i = model_params[i]
        
        for j in range(num_models):
            if i == j:
                continue

            t_pair_start = time.time()
            name_j = eval_model_names[j]
            chunk_size = 50
            total_ll = 0.0
            num_seqs = len(model_xs[j])
            for start_idx in range(0, num_seqs, chunk_size):
                end_idx = min(start_idx + chunk_size, num_seqs)
                chunk_x = jnp.array(model_xs[j][start_idx:end_idx])
                chunk_mask = mask_gpu[start_idx:end_idx]
                chunk_val = marginal_log_likelihood(
                    chunk_mask,
                    chunk_x,
                    params_i["Ab"],
                    params_i["Q"],
                    params_i["pi"]
                ).block_until_ready()
                total_ll += float(chunk_val.item())
            
            mlls[i, j] = total_ll
            pair_dur = time.time() - t_pair_start
            pair_times.append(pair_dur)
            completed_pairs += 1

            avg_time = np.mean(pair_times)
            rem_pairs = total_pairs - completed_pairs
            eta_min = (rem_pairs * avg_time) / 60.0

            print(
                f"  [{completed_pairs:3d}/{total_pairs:3d}] "
                f"P(x[{name_j[-6:]}] | theta[{name_i[-6:]}]) = {mlls[i, j]:12.2f} "
                f"({pair_dur:.2f}s | avg: {avg_time:.2f}s | ETA: {eta_min:.1f}m)"
            )

    total_eml_time = time.time() - eml_start_time
    print(f"\nAll {total_pairs} cross-likelihood pairs evaluated in {total_eml_time:.1f}s ({total_eml_time / 60.0:.2f} min).")

    # 5. Compute EML Scores and Standard Errors
    scores = mlls.sum(axis=1) / (num_models - 1)
    variances = (mlls**2).sum(axis=1) / (num_models - 1) - scores**2
    variances = np.maximum(variances, 0.0)
    standard_errors = np.sqrt(variances / (num_models - 1))

    sorted_indices = np.argsort(scores)[::-1]
    best_idx = sorted_indices[0]
    best_model_name = eval_model_names[best_idx]
    
    print("\n" + "=" * 80)
    print("  PHASE 4: FINAL 8-KEYPOINT MODEL RANKINGS (EXPECTED MARGINAL LIKELIHOOD)")
    print("=" * 80)
    print(f"  Best Model: {best_model_name} (Score: {scores[best_idx]:.2f} +/- {standard_errors[best_idx]:.2f})\n")

    ranking_data = []
    for rank, idx in enumerate(sorted_indices, 1):
        name = eval_model_names[idx]
        score = scores[idx]
        stderr = standard_errors[idx]
        
        sum_path = PROJECT_DIR / name / "model_summary.json"
        summary_info = {}
        if sum_path.exists():
            with open(sum_path) as sf:
                summary_info = json.load(sf)
                
        seed = summary_info.get("seed", "N/A")
        med_dur = summary_info.get("median_duration_frames", "N/A")
        wall_time = summary_info.get("wall_clock_seconds", 0)

        ranking_data.append({
            "rank": rank,
            "model_name": name,
            "eml_score": score,
            "standard_error": stderr,
            "seed": seed,
            "iterations": 200,
            "median_duration_frames": med_dur,
            "training_wall_seconds": round(wall_time, 1)
        })

        print(
            f"  #{rank:02d}: {name} (Seed {seed}) | "
            f"EML: {score:12.2f} +/- {stderr:9.2f} | "
            f"Med Dur: {med_dur} frames | Train: {wall_time/60.0:.1f}m"
        )

    # 6. Save CSV
    csv_path = OUTPUT_DIR / "eml_rankings_8kp.csv"
    with open(csv_path, "w") as f:
        f.write("rank,model_name,eml_score,standard_error,seed,iterations,median_duration_frames,training_wall_seconds\n")
        for row in ranking_data:
            f.write(
                f"{row['rank']},{row['model_name']},{row['eml_score']:.4f},"
                f"{row['standard_error']:.4f},{row['seed']},{row['iterations']},"
                f"{row['median_duration_frames']},{row['training_wall_seconds']}\n"
            )
    print(f"\nSaved rankings CSV to: {csv_path}")

    # 7. Generate Visualizations (EML Plot & Heatmap)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    ranked_names = [r["model_name"] for r in ranking_data]
    ranked_scores = [r["eml_score"] for r in ranking_data]
    ranked_errs = [r["standard_error"] for r in ranking_data]

    y_pos = np.arange(len(ranked_names))
    ax.errorbar(ranked_scores, y_pos, xerr=ranked_errs, fmt='o', color='#1f77b4',
                ecolor='#aec7e8', elinewidth=2, capsize=4, markersize=8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(ranked_names, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel("Expected Marginal Log-Likelihood (EML)", fontsize=11)
    ax.set_title("8-Keypoint Models: Ranked by EML (Higher is Better)", fontsize=12, fontweight='bold')
    ax.grid(axis='x', linestyle='--', alpha=0.6)
    ax.plot(ranked_scores[0], y_pos[0], 'o', color='#2ca02c', markersize=10, label=f"Best: {ranked_names[0]}")
    ax.legend(loc="lower right")

    ax2 = axes[1]
    reordered_mlls = mlls[sorted_indices, :][:, sorted_indices]
    cax = ax2.imshow(reordered_mlls, cmap="viridis", aspect="auto")
    fig.colorbar(cax, ax=ax2, label="log P(x^j | theta^i)")
    ax2.set_xticks(np.arange(num_models))
    ax2.set_yticks(np.arange(num_models))
    ax2.set_xticklabels([n[-6:] for n in ranked_names], fontsize=10)
    ax2.set_yticklabels([n[-6:] for n in ranked_names], fontsize=10)
    ax2.set_xlabel("Continuous Latent Source (x^j)", fontsize=11)
    ax2.set_ylabel("Parameter Model (theta^i)", fontsize=11)
    ax2.set_title("Cross-Marginal Likelihood Matrix", fontsize=12, fontweight='bold')

    plt.tight_layout()
    plot_path_png = OUTPUT_DIR / "eml_scores_8kp.png"
    plt.savefig(plot_path_png, dpi=300)
    plt.close()
    print(f"Saved EML ranking plot to {plot_path_png}")

    # 8. Report
    report_path = OUTPUT_DIR / "EML_8KP_MODEL_SELECTION_REPORT.md"
    with open(report_path, "w") as f:
        f.write(f"""# 8-Keypoint MoSeq Model Selection: EML Benchmark Report

**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}  
**Project Directory**: `{PROJECT_DIR}`  
**Hardware Accelerator**: NVIDIA GeForce RTX 4090  

---

## 1. Top-Ranked Model Summary

* **Selected Winning Model**: **`{best_model_name}`**
* **Seed**: `{ranking_data[0]['seed']}`
* **EML Score**: **`{ranking_data[0]['eml_score']:.2f} +/- {ranking_data[0]['standard_error']:.2f}`**
* **Median Syllable Duration**: `{ranking_data[0]['median_duration_frames']} frames` (~{ranking_data[0]['median_duration_frames']/40.0*1000:.0f} ms)
* **Results Path**: `{PROJECT_DIR / best_model_name / "results.h5"}`

---

## 2. Complete Model Rankings

| Rank | Model Name | Seed | EML Score | Std Error | Median Duration | Training Wall Time |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
""")
        for r in ranking_data:
            f.write(
                f"| **#{r['rank']:02d}** | `{r['model_name']}` | `{r['seed']}` | "
                f"**{r['eml_score']:.2f}** | +/- {r['standard_error']:.2f} | "
                f"{r['median_duration_frames']} frames ({r['median_duration_frames']/40.0*1000:.0f} ms) | "
                f"{r['training_wall_seconds']/60.0:.1f} min |\n"
            )

    send_ntfy(
        f"8KP EML Ranking Complete: Best = {best_model_name}",
        f"Top-ranked 8KP model is {best_model_name} (Seed {ranking_data[0]['seed']}) with EML {ranking_data[0]['eml_score']:.2f}.",
        "trophy",
        "high"
    )

    print("\n" + "=" * 80)
    print(f"  EML EVALUATION COMPLETE: WINNER IS {best_model_name}")
    print("=" * 80)

if __name__ == "__main__":
    main()
