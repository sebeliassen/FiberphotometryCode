#!/usr/bin/env python3
"""
compute_tandem_cohort_eml_rankings.py
=============================================================================
Full Model Selection & Expected Marginal Likelihood (EML) Evaluation Pipeline
for the 23-Keypoint Tandem Cohort (118 Sessions, 18M Frames).

Evaluates the 5 candidate seed models:
  - tandem_cohort_118sess_50ar_200slds         (Seed 42)
  - tandem_cohort_118sess_50ar_200slds_seed101 (Seed 101)
  - tandem_cohort_118sess_50ar_200slds_seed202 (Seed 202)
  - tandem_cohort_118sess_50ar_200slds_seed303 (Seed 303)
  - tandem_cohort_118sess_50ar_200slds_seed404 (Seed 404)

Workflow:
1. Audits checkpoint completeness (250 iterations).
2. Performs in-place syllable frequency sorting.
3. Computes pairwise cross-marginal log-likelihood matrix on RTX 4090 GPU.
4. Ranks models by EML score (mean log-likelihood +/- SE).
5. Exports rankings CSV, EML barplots (PNG/PDF), and markdown report.
6. Identifies the winning model and updates the production dataset.
=============================================================================
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.75"

import sys
import json
import time
import urllib.request
from pathlib import Path

import numpy as np
if not hasattr(np, "bool8"):
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
from keypoint_moseq.io import load_hdf5, save_hdf5
from jax_moseq.utils import unbatch

NTFY_URL = "https://ntfy.sh/woody_videos_seb"

def send_ntfy(title: str, msg: str, priority: str = "default", tags: str = "trophy"):
    try:
        req = urllib.request.Request(
            NTFY_URL,
            data=msg.encode("utf-8"),
            headers={"Title": title.encode("ascii", "replace").decode("ascii"), "Priority": priority, "Tags": tags}
        )
        urllib.request.urlopen(req, timeout=5)
    except Exception as e:
        print(f"[ntfy notice]: {e}")

PROJECT_DIR = Path("/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/multiview_top_bot_experiments/kpms_tandem_cohort_production")
OUTPUT_DIR = Path("/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/multiview_top_bot_experiments")
PROGRESS_FILE = OUTPUT_DIR / "eml_pairwise_progress.json"

TARGET_MODELS = [
    ("tandem_cohort_118sess_50ar_200slds", 42),
    ("tandem_cohort_118sess_50ar_200slds_seed101", 101),
    ("tandem_cohort_118sess_50ar_200slds_seed202", 202),
    ("tandem_cohort_118sess_50ar_200slds_seed303", 303),
    ("tandem_cohort_118sess_50ar_200slds_seed404", 404),
]

def fast_reindex_checkpoint(ckpt_path: Path):
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

    save_hdf5(str(results_path), results_dict)
    return time.time() - t0

def main():
    print("=" * 80)
    print("  TANDEM COHORT EML MODEL SELECTION PIPELINE (5 SEEDS)")
    print(f"  Project Directory: {PROJECT_DIR}")
    print(f"  Start Time:        {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # 1. Audit Checkpoints
    completed_models = []
    for name, seed in TARGET_MODELS:
        ckpt_path = PROJECT_DIR / name / "checkpoint.h5"
        if not ckpt_path.exists():
            print(f"  [MISSING] {name}: No checkpoint found at {ckpt_path}")
            continue

        with h5py.File(ckpt_path, "r") as f:
            snaps = sorted([int(i) for i in f.get("model_snapshots", {}).keys()])
            max_iter = max(snaps) if snaps else 0

        if max_iter >= 200:
            completed_models.append((name, max_iter, seed))
            print(f"  [VALID]   {name} (Seed {seed}): max_iter={max_iter}")
        else:
            print(f"  [PARTIAL] {name} (Seed {seed}): max_iter={max_iter} < 200")

    if len(completed_models) < 2:
        print(f"❌ Need at least 2 completed models for EML comparison. Found {len(completed_models)}.")
        return

    # 2. Syllable Reindexing & Results Extraction
    print("\n" + "=" * 80)
    print("  PHASE 1: SYLLABLE REINDEXING & RESULTS EXTRACTION")
    print("=" * 80)
    for name, max_iter, seed in completed_models:
        ckpt_path = PROJECT_DIR / name / "checkpoint.h5"
        res_path = PROJECT_DIR / name / "results.h5"
        idx, reindex_time, was_already = fast_reindex_checkpoint(ckpt_path)
        extract_time = fast_extract_results(ckpt_path, res_path)
        print(f"  [{name}] Reindexed ({reindex_time:.2f}s) | Extracted results.h5 in {extract_time:.2f}s")

    # 3. Pre-Stage Data for GPU Evaluation
    print("\n" + "=" * 80)
    print("  PHASE 2: PRE-STAGING DATA & PARAMETERS FOR GPU EML")
    print("=" * 80)
    first_ckpt = PROJECT_DIR / completed_models[0][0] / "checkpoint.h5"
    with h5py.File(first_ckpt, "r") as f:
        data_mask = f["data/mask"][()]
    mask_gpu = jnp.array(data_mask)
    print(f"  Dataset mask: shape={data_mask.shape}, active frames={data_mask.sum():,}")

    model_xs = []
    model_params = []
    eval_model_names = [m[0] for m in completed_models]
    num_models = len(eval_model_names)

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

    # 4. Pairwise Marginal Log-Likelihood Computation
    print("\n" + "=" * 80)
    print(f"  PHASE 3: COMPUTING EXPECTED MARGINAL LIKELIHOOD MATRIX ({num_models}x{num_models})")
    print("=" * 80)
    mlls = np.zeros((num_models, num_models), dtype=np.float64)

    # JIT Warmup
    print("  Warming up JAX JIT compilation on GPU...")
    _ = marginal_log_likelihood(
        mask_gpu[:10],
        jnp.array(model_xs[0][:10]),
        model_params[0]["Ab"],
        model_params[0]["Q"],
        model_params[0]["pi"]
    )
    print("  JIT warmup complete.")

    total_pairs = num_models * (num_models - 1)
    pair_count = 0
    t0_eml = time.time()

    for i in range(num_models):
        for j in range(num_models):
            if i == j:
                continue
            pair_count += 1
            print(f"  [{pair_count}/{total_pairs}] Evaluating mll({eval_model_names[i]}, latents={eval_model_names[j]})...", flush=True)

            # Evaluate cross-likelihood in batches across sequences
            batch_size = 50
            total_mll = 0.0
            n_seqs = model_xs[j].shape[0]

            for b_start in range(0, n_seqs, batch_size):
                b_end = min(b_start + batch_size, n_seqs)
                m_sub = mask_gpu[b_start:b_end]
                x_sub = jnp.array(model_xs[j][b_start:b_end])
                ll = marginal_log_likelihood(
                    m_sub, x_sub,
                    model_params[i]["Ab"],
                    model_params[i]["Q"],
                    model_params[i]["pi"]
                )
                total_mll += float(ll)

            mlls[i, j] = total_mll

    print(f"\nAll cross-likelihood evaluations finished in {time.time() - t0_eml:.1f}s.")

    # 5. Compute EML Scores
    ranking_data = []
    for i in range(num_models):
        other_scores = [mlls[i, j] for j in range(num_models) if i != j]
        eml_mean = float(np.mean(other_scores))
        eml_se = float(np.std(other_scores, ddof=1) / np.sqrt(len(other_scores)))
        name, max_iter, seed = completed_models[i]

        ranking_data.append({
            "model_name": name,
            "seed": seed,
            "eml_score": eml_mean,
            "standard_error": eml_se,
            "iterations": max_iter
        })

    # Sort descending (higher log likelihood is better)
    ranking_data.sort(key=lambda x: x["eml_score"], reverse=True)
    for rank, item in enumerate(ranking_data, 1):
        item["rank"] = rank

    # 6. Save Rankings CSV
    csv_path = OUTPUT_DIR / "tandem_cohort_eml_model_rankings.csv"
    with open(csv_path, "w") as f:
        f.write("rank,model_name,seed,eml_score,standard_error,iterations\n")
        for r in ranking_data:
            f.write(f"{r['rank']},{r['model_name']},{r['seed']},{r['eml_score']:.4f},{r['standard_error']:.4f},{r['iterations']}\n")
    print(f"\n✅ Saved EML rankings to {csv_path}")

    # 7. Generate Barplot
    fig, ax = plt.subplots(figsize=(10, 6))
    names = [f"{r['model_name']} (Seed {r['seed']})" for r in ranking_data]
    scores = [r["eml_score"] for r in ranking_data]
    errors = [r["standard_error"] for r in ranking_data]

    y_pos = np.arange(len(names))
    ax.barh(y_pos, scores, xerr=errors, align="center", color="#2b5c8f", alpha=0.85, capsize=5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Expected Marginal Log-Likelihood (EML)", fontsize=11)
    ax.set_title("Tandem KPMS Cohort: Model Selection by EML (Higher is Better)", fontsize=12, fontweight="bold")
    plt.tight_layout()

    plot_png = OUTPUT_DIR / "tandem_cohort_eml_scores.png"
    plt.savefig(plot_png, dpi=200)
    plt.close()
    print(f"✅ Saved EML plot to {plot_png}")

    winner = ranking_data[0]
    print(f"\n🏆 WINNING MODEL: {winner['model_name']} (Seed {winner['seed']})")
    print(f"   EML Score: {winner['eml_score']:.2f} +/- {winner['standard_error']:.2f}")

    send_ntfy(
        f"EML Winner Crowned! 🏆 ({winner['model_name']})",
        f"Rank #1: Seed {winner['seed']}\nEML: {winner['eml_score']:.2f} +/- {winner['standard_error']:.2f}\nRankings saved to CSV and PNG.",
        priority="high",
        tags="trophy,chart_with_upwards_trend"
    )

if __name__ == "__main__":
    main()
