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
2. Computes syllable frequency re-indexing IN MEMORY (source checkpoints are
   opened read-only and never modified); writes derived, manifest-verified
   results_reindexed.h5 with the old->new label map.
3. Validates that candidate models share dataset/mask/shape identity, then computes
   pairwise cross-marginal log-likelihood matrix on RTX 4090 GPU.
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

import argparse
import re
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

import h5py

EXP_DIR = Path(__file__).resolve().parent.parent
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

from fpcode.kpms import evaluation as ea
from fpcode.common.cache_manifest import (
    OK as CACHE_OK,
)
from fpcode.common.cache_manifest import (
    atomic_path,
    verify_manifest,
    write_manifest,
)

# Heavy dependencies (JAX / KPMS / matplotlib) are imported lazily by
# _load_heavy_deps() so that --help and --dry-run never initialize JAX or the GPU.
jax = jnp = plt = None
marginal_log_likelihood = get_frequencies = load_hdf5 = save_hdf5 = unbatch = None


def _load_heavy_deps():
    global jax, jnp, plt, marginal_log_likelihood, get_frequencies, load_hdf5, save_hdf5, unbatch
    import jax as _jax
    import jax.numpy as _jnp
    import matplotlib
    matplotlib.use("Agg")
    import keypoint_moseq  # noqa: F401  (registers KPMS / jax_moseq)
    import matplotlib.pyplot as _plt
    from jax_moseq.utils import unbatch as _ub
    from keypoint_moseq.fitting import marginal_log_likelihood as _mll
    from keypoint_moseq.io import load_hdf5 as _lh
    from keypoint_moseq.io import save_hdf5 as _sh
    from keypoint_moseq.util import get_frequencies as _gf
    jax, jnp, plt = _jax, _jnp, _plt
    marginal_log_likelihood, get_frequencies = _mll, _gf
    load_hdf5, save_hdf5, unbatch = _lh, _sh, _ub

from fpcode.common.notify import send_ntfy
from fpcode.common.paths import REPO_ROOT

DEFAULT_PROJECT_DIR = REPO_ROOT / "syllable_generation/multiview_top_bot_experiments/kpms_tandem_cohort_continuous_calibrated"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "syllable_generation/multiview_top_bot_experiments/04_reports_and_figures"

DERIVED_RESULTS_NAME = "results_reindexed.h5"
LABEL_POLICY = "stable_frequency_order_v1"
EXTRACTION_VERSION = 1  # bump when extraction logic below changes (invalidates derived caches)


def compute_reindex(ckpt_path: Path):
    """READ-ONLY syllable frequency re-indexing.

    Returns ``(index, elapsed_s, is_identity)`` where new label ``i`` is old state
    ``index[i]``. The checkpoint is opened ``"r"``; nothing is written. Uses a stable
    sort (ties keep original order), so the mapping is idempotent.
    """
    t0 = time.time()
    ident = ea.read_checkpoint_identity(ckpt_path)
    it = ident["iteration"]
    with h5py.File(ckpt_path, "r") as f:
        z = f[f"model_snapshots/{it}/states/z"][()]
        bounds = f["metadata/arr1"][()]
    mask = ea.reconstruct_mask(bounds, ident["seg_len"])
    freqs = get_frequencies(z, mask[:, ident["nlags"]:], ident["num_states"], runlength=True)
    index = ea.stable_frequency_order(freqs)
    return index, time.time() - t0, bool(np.array_equal(index, np.arange(len(index))))


def _results_config(ident: dict, index) -> dict:
    return {
        "extraction_version": EXTRACTION_VERSION,
        "label_policy": LABEL_POLICY,
        "iteration": ident["iteration"],
        "num_states": ident["num_states"],
        "metadata_hash": ident["metadata_hash"],
        "label_mapping_hash": ea.mapping_hash(index),
    }


def fast_extract_results(ckpt_path: Path, results_path: Path, index):
    """Write a DERIVED results file with syllables relabeled in memory.

    The source checkpoint is read-only. The file is published atomically and gets a
    manifest keyed on the checkpoint identity (size/mtime), iteration, dataset
    metadata and label mapping, so results from a different checkpoint state,
    iteration or label mapping are never reused.
    """
    import gc
    t0 = time.time()
    ident = ea.read_checkpoint_identity(ckpt_path)
    config = _results_config(ident, index)
    status, reasons = verify_manifest(
        results_path, kind="results_h5", inputs=[ckpt_path], config=config,
        code_files=[Path(ea.__file__)],
    )
    if status == CACHE_OK:
        print(f"  [CACHE HIT] {results_path.name} verified against checkpoint + label mapping. Skipping extraction.")
        return 0.0
    if results_path.exists():
        print(f"  [CACHE {status}] {results_path.name}: {'; '.join(reasons)} -> rebuilding atomically.")

    it = ident["iteration"]
    inv = ea.invert_permutation(index)
    with h5py.File(ckpt_path, "r") as f:
        x = f[f"model_snapshots/{it}/states/x"][()]
        z = f[f"model_snapshots/{it}/states/z"][()]
        v = f[f"model_snapshots/{it}/states/v"][()]
        h = f[f"model_snapshots/{it}/states/h"][()]
        keys = f["metadata/arr0"][()]
        bounds = f["metadata/arr1"][()]
    metadata = (keys, bounds)

    nlags = x.shape[1] - z.shape[1]
    z_padded = np.pad(ea.relabel_states(z, inv), ((0, 0), (nlags, 0)), mode="edge")
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

    with atomic_path(results_path) as tmp:
        save_hdf5(str(tmp), results_dict)
        del x, z, v, h, z_padded, syllables, latent_state, centroid, heading, results_dict
        gc.collect()
        try:
            with open(tmp, "rb") as f_res:
                os.posix_fadvise(f_res.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        except Exception:
            pass
    write_manifest(
        results_path, kind="results_h5", inputs=[ckpt_path], config=config,
        code_files=[Path(ea.__file__)],
        extra={
            "source_checkpoint": str(ckpt_path),
            "index_new_to_old": [int(i) for i in index],
            "old_to_new": {str(k): v for k, v in ea.old_to_new_map(index).items()},
            "identity_mapping": bool(np.array_equal(index, np.arange(len(index)))),
        },
    )
    return time.time() - t0

def main():
    parser = argparse.ArgumentParser(description="Tandem KPMS Cohort EML Model Selection Pipeline")
    parser.add_argument(
        "--project-dir",
        type=str,
        default=str(DEFAULT_PROJECT_DIR),
        help="Path to KPMS project containing candidate model subdirectories",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to save EML rankings CSV and plot",
    )
    parser.add_argument(
        "--min-iters",
        type=int,
        default=200,
        help="Minimum completed iterations required for a candidate model",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="*",
        default=None,
        help="Explicit list of model directory names to evaluate",
    )
    parser.add_argument(
        "--output-tag",
        type=str,
        default=None,
        help="Suffix for output filenames (default: derived from project dir name, "
             "e.g. 'continuous_calibrated'). Prevents clobbering other cohorts' results.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing ranking CSV/PNG outputs with the same tag",
    )
    parser.add_argument(
        "--skip-results",
        action="store_true",
        help="Skip writing derived results_reindexed.h5 (EML itself does not need it)",
    )
    parser.add_argument(
        "--allow-mixed-iterations",
        action="store_true",
        help="Permit comparing models evaluated at different checkpoint iterations",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Audit models, validate comparability and print the plan; no JAX/GPU, no writes",
    )
    args = parser.parse_args()

    project_dir = Path(args.project_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "audits").mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("  TANDEM COHORT EML MODEL SELECTION PIPELINE")
    print(f"  Project Directory: {project_dir}")
    print(f"  Output Directory:  {output_dir}")
    print(f"  Start Time:        {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # 1. Audit Checkpoints
    if args.models:
        target_models = []
        for m_name in args.models:
            m = re.search(r"seed(\d+)", m_name)
            seed = int(m.group(1)) if m else 42
            target_models.append((m_name, seed))
    else:
        # Auto-discover models
        target_models = []
        for d in sorted(project_dir.iterdir()):
            if d.is_dir() and (d / "checkpoint.h5").exists() and d.name.startswith("tandem_"):
                m = re.search(r"seed(\d+)", d.name)
                seed = int(m.group(1)) if m else (42 if "seed" not in d.name else 0)
                target_models.append((d.name, seed))

    print(f"  Found {len(target_models)} candidate model folders to audit.")

    completed_models = []
    for name, seed in target_models:
        ckpt_path = project_dir / name / "checkpoint.h5"
        if not ckpt_path.exists():
            print(f"  [MISSING] {name}: No checkpoint found at {ckpt_path}")
            continue

        with h5py.File(ckpt_path, "r") as f:
            snaps = sorted([int(i) for i in f.get("model_snapshots", {}).keys()])
            max_iter = max(snaps) if snaps else 0

        if max_iter >= args.min_iters:
            completed_models.append((name, max_iter, seed))
            print(f"  [VALID]   {name} (Seed {seed}): max_iter={max_iter}")
        else:
            print(f"  [PARTIAL] {name} (Seed {seed}): max_iter={max_iter} < {args.min_iters}")

    if len(completed_models) < 2:
        print(f"❌ Need at least 2 completed models for EML comparison. Found {len(completed_models)}.")
        return

    # 1b. Validate that candidates share dataset / mask / shape identity BEFORE any EML.
    ckpt_paths = [project_dir / name / "checkpoint.h5" for name, _, _ in completed_models]
    identities = {
        name: ea.read_checkpoint_identity(project_dir / name / "checkpoint.h5", iteration=max_iter)
        for name, max_iter, _ in completed_models
    }
    try:
        ea.validate_model_compatibility(identities, require_same_iteration=not args.allow_mixed_iterations)
    except ea.IncompatibleModelsError as e:
        print(f"❌ {e}")
        sys.exit(2)
    ref = next(iter(identities.values()))
    print(f"  [COMPATIBLE] {len(identities)} models share dataset identity: "
          f"{ref['n_seq']} segments x {ref['seg_len']} frames, {ref['active_frames']:,} active frames, "
          f"latent_dim={ref['latent_dim']}, metadata_hash={ref['metadata_hash'][:12]}")

    if args.dry_run:
        print("\n  [DRY RUN] Plan (nothing written, JAX/GPU not initialized):")
        for name, max_iter, seed in completed_models:
            print(f"    - {name} (seed {seed}, iter {max_iter}): source checkpoint opened READ-ONLY")
            if not args.skip_results:
                print(f"        derived results -> {project_dir / name / DERIVED_RESULTS_NAME}")
        n = len(completed_models)
        print(f"    - {n * (n - 1)} pairwise cross-likelihood evaluations on GPU")
        print(f"    - rankings/plot tagged '{args.output_tag or project_dir.name.replace('kpms_tandem_cohort_', '')}' in {output_dir}")
        return

    _load_heavy_deps()
    sig_before = ea.stat_signature(ckpt_paths)

    # 2. READ-ONLY syllable re-indexing & derived results extraction
    print("\n" + "=" * 80)
    print("  PHASE 1: READ-ONLY SYLLABLE REINDEXING & DERIVED RESULTS EXTRACTION")
    print("=" * 80)
    for name, max_iter, seed in completed_models:
        ckpt_path = project_dir / name / "checkpoint.h5"
        res_path = project_dir / name / DERIVED_RESULTS_NAME
        idx, reindex_time, was_identity = compute_reindex(ckpt_path)
        n_moved = int((idx != np.arange(len(idx))).sum())
        if args.skip_results:
            extract_msg = "results extraction skipped"
        else:
            extract_time = fast_extract_results(ckpt_path, res_path, idx)
            extract_msg = f"derived {DERIVED_RESULTS_NAME} in {extract_time:.2f}s"
        print(f"  [{name}] reindex map computed in-memory ({reindex_time:.2f}s, {n_moved} labels move) | {extract_msg}")

    # 3. Pre-Stage Data for GPU Evaluation
    print("\n" + "=" * 80)
    print("  PHASE 2: PRE-STAGING DATA & PARAMETERS FOR GPU EML")
    print("=" * 80)
    first_ckpt = project_dir / completed_models[0][0] / "checkpoint.h5"
    with h5py.File(first_ckpt, "r") as f:
        if "data/mask" in f:
            data_mask = f["data/mask"][()]
        else:
            meta = load_hdf5(str(first_ckpt), "metadata")
            keys, bounds = meta
            seg_len = f[f"model_snapshots/{completed_models[0][1]}/states/x"].shape[1]
            data_mask = np.zeros((len(bounds), seg_len), dtype=bool)
            for bi, (s_b, e_b) in enumerate(bounds):
                data_mask[bi, :(e_b - s_b)] = True

    mask_gpu = jnp.array(data_mask)
    print(f"  Dataset mask: shape={data_mask.shape}, active frames={data_mask.sum():,}")

    model_params = []
    eval_model_names = [m[0] for m in completed_models]
    num_models = len(eval_model_names)

    print("  Loading parameter matrices (Ab, Q, pi) for all candidate models...")
    for name, max_iter, seed in completed_models:
        ckpt_path = project_dir / name / "checkpoint.h5"
        with h5py.File(ckpt_path, "r") as f:
            snap = f[f"model_snapshots/{max_iter}"]
            Ab = snap["params/Ab"][()]
            Q = snap["params/Q"][()]
            pi = snap["params/pi"][()]

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
    first_ckpt = project_dir / completed_models[0][0] / "checkpoint.h5"
    with h5py.File(first_ckpt, "r") as f:
        warmup_x = f[f"model_snapshots/{completed_models[0][1]}/states/x"][:10]
    _ = marginal_log_likelihood(
        mask_gpu[:10],
        jnp.array(warmup_x),
        model_params[0]["Ab"],
        model_params[0]["Q"],
        model_params[0]["pi"]
    )
    del warmup_x
    print("  JIT warmup complete.")

    total_pairs = num_models * (num_models - 1)
    pair_count = 0
    t0_eml = time.time()

    # Outer loop streams ONE model's latent states at a time
    for j in range(num_models):
        lat_name, lat_max_iter, lat_seed = completed_models[j]
        ckpt_path = project_dir / lat_name / "checkpoint.h5"
        print(f"\n  [Latent Pool {j+1}/{num_models}] Loading latent trajectories for {lat_name}...", flush=True)
        with h5py.File(ckpt_path, "r") as f:
            x_j = f[f"model_snapshots/{lat_max_iter}/states/x"][()]

        n_seqs = x_j.shape[0]
        batch_size = 50

        for i in range(num_models):
            if i == j:
                continue
            pair_count += 1
            print(f"    [{pair_count}/{total_pairs}] Evaluating mll(model={eval_model_names[i]}, latents={eval_model_names[j]})...", flush=True)

            total_mll = 0.0
            for b_start in range(0, n_seqs, batch_size):
                b_end = min(b_start + batch_size, n_seqs)
                m_sub = mask_gpu[b_start:b_end]
                x_sub = jnp.array(x_j[b_start:b_end])
                ll = marginal_log_likelihood(
                    m_sub, x_sub,
                    model_params[i]["Ab"],
                    model_params[i]["Q"],
                    model_params[i]["pi"]
                )
                total_mll += float(ll)

            mlls[i, j] = total_mll

        del x_j
        import gc
        gc.collect()

    print(f"\nAll cross-likelihood evaluations finished in {time.time() - t0_eml:.1f}s.")

    ea.assert_unchanged(sig_before, ea.stat_signature(ckpt_paths))
    print(f"  [IMMUTABLE] {len(ckpt_paths)} source checkpoints verified unchanged (size + mtime_ns).")

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

    # 6. Save Rankings CSV (tagged per project; never clobber other cohorts' results)
    tag = args.output_tag or project_dir.name.replace("kpms_tandem_cohort_", "")
    csv_paths = [
        output_dir / f"tandem_cohort_eml_model_rankings_{tag}.csv",
        output_dir / f"audits/tandem_cohort_eml_model_rankings_{tag}.csv",
    ]
    plot_png = output_dir / f"tandem_cohort_eml_scores_{tag}.png"
    if not args.overwrite:
        existing = [p for p in csv_paths + [plot_png] if p.exists()]
        if existing:
            alt = time.strftime("%Y%m%d_%H%M%S")
            tag = f"{tag}_{alt}"
            csv_paths = [
                output_dir / f"tandem_cohort_eml_model_rankings_{tag}.csv",
                output_dir / f"audits/tandem_cohort_eml_model_rankings_{tag}.csv",
            ]
            plot_png = output_dir / f"tandem_cohort_eml_scores_{tag}.png"
            print(f"⚠️  Existing outputs found ({len(existing)}); writing to new tag '{tag}' instead of overwriting.")
    for csv_path in csv_paths:
        with open(csv_path, "w") as f:
            f.write("rank,model_name,seed,eml_score,standard_error,iterations\n")
            for r in ranking_data:
                f.write(f"{r['rank']},{r['model_name']},{r['seed']},{r['eml_score']:.4f},{r['standard_error']:.4f},{r['iterations']}\n")
        print(f"✅ Saved EML rankings to {csv_path}")

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

