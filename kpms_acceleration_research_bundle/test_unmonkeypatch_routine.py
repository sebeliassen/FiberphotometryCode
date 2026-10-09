#!/usr/bin/env python3
"""
diagnostics/test_unmonkeypatch_routine.py
=============================================================================
Controlled Multi-Arm Diagnostic Routine:
Systematically "un-monkeypatches" components of the KPMS pipeline to determine:
1. Does the median syllable duration decrease during SLDS fitting under 100%
   unpatched upstream JAX as well?
2. Does the monkeypatched pipeline match the unpatched baseline's duration,
   mean duration, and active syllable count at every iteration?
3. Pinpoints whether any individual kernel (continuous Banded Cholesky vs JAX Kalman,
   Numba FFBS vs JAX FFBS, AR params, etc.) alters duration dynamics.
=============================================================================
"""

import sys
import os
import time
import tempfile
import numpy as np

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

REPO_ROOT = "/home/fsp585/Desktop/FiberphotometryCode"
KPMS_DIR = os.path.join(REPO_ROOT, "syllable_generation", "syllabizers", "keypoint_moseq_syll")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if KPMS_DIR not in sys.path:
    sys.path.insert(0, KPMS_DIR)

import jax
import jax.numpy as jnp
import jax.random as jr
import keypoint_moseq as kpms
from jax_moseq.models import keypoint_slds
import utils.resample_batching as rb


def compute_syllable_stats(z_seqs):
    """Compute run-length encoded syllable statistics."""
    bouts = []
    active_sylls = set()
    for seq in z_seqs:
        seq = np.asarray(seq)
        if len(seq) == 0:
            continue
        active_sylls.update(seq.tolist())
        # Run-length encode
        diffs = np.diff(seq) != 0
        idx = np.where(diffs)[0] + 1
        starts = np.r_[0, idx]
        ends = np.r_[idx, len(seq)]
        lens = ends - starts
        bouts.extend(lens.tolist())
    
    bouts = np.array(bouts)
    if len(bouts) == 0:
        return {"median": 0, "mean": 0, "p10": 0, "p25": 0, "p75": 0, "p90": 0, "active": 0, "n_bouts": 0}
    
    return {
        "median": float(np.median(bouts)),
        "mean": float(np.mean(bouts)),
        "p10": float(np.percentile(bouts, 10)),
        "p25": float(np.percentile(bouts, 25)),
        "p75": float(np.percentile(bouts, 75)),
        "p90": float(np.percentile(bouts, 90)),
        "active": len(active_sylls),
        "n_bouts": len(bouts)
    }


def generate_synthetic_session_data(seed=42, N=4, T=1200, K=17, D=10):
    """Generate realistic synthetic keypoint dataset with persistent bouts."""
    rng = np.random.default_rng(seed)
    coords = {}
    confs = {}
    bodyparts = [f"bp_{i}" for i in range(K)]

    for i in range(N):
        key = f"synthetic_mouse_{i}"
        # Generate smooth walking trajectory with persistent behavioral states
        t = np.linspace(0, 4 * np.pi, T)
        base_x = 200 + 100 * np.cos(t * (0.5 + 0.1 * i))
        base_y = 200 + 100 * np.sin(t * (0.5 + 0.1 * i))
        
        c = np.zeros((T, K, 2), dtype=np.float64)
        for bp in range(K):
            offset_x = (bp - K // 2) * 5.0
            offset_y = (bp % 3 - 1) * 3.0
            c[:, bp, 0] = base_x + offset_x + rng.normal(0, 0.5, size=T)
            c[:, bp, 1] = base_y + offset_y + rng.normal(0, 0.5, size=T)
        
        conf = rng.uniform(0.85, 1.0, size=(T, K))
        coords[key] = c
        confs[key] = conf

    return coords, confs, bodyparts


def run_experiment():
    print("=" * 80)
    print("  KPMS DIAGNOSTIC ROUTINE: MONKEYPATCHED VS UNPATCHED DURATION AUDIT")
    print("=" * 80)

    with tempfile.TemporaryDirectory() as tmp_dir:
        # 1. Setup project
        kpms.setup_project(tmp_dir, overwrite=True)
        config = kpms.load_config(tmp_dir)
        config["fps"] = 60
        config["trans_hypparams"]["kappa"] = 1e6
        config["trans_hypparams"]["num_states"] = 30
        config["ar_hypparams"]["latent_dim"] = 10
        kpms.update_config(tmp_dir, **config)

        # 2. Format synthetic data
        N_SEQS, T_LEN = 4, 1200
        coords, confs, bodyparts = generate_synthetic_session_data(seed=123, N=N_SEQS, T=T_LEN)
        config["bodyparts"] = bodyparts
        config["use_bodyparts"] = bodyparts
        kpms.update_config(tmp_dir, **config)
        data, metadata = kpms.format_data(coords, confs, seg_length=T_LEN, max_seg_length=T_LEN, **config)
        data["Y"] = np.asarray(data["Y"])

        pca = kpms.fit_pca(**data, **config)
        kpms.save_pca(pca, tmp_dir)

        # 3. Fit shared base AR model for 10 iterations so all test arms start identically
        print("\n[Step 1/3] Initializing and pre-fitting base AR model (10 iters) on shared seed...")
        rb.restore_original_kernels()  # start clean unpatched
        base_seed = jr.PRNGKey(42)
        base_model = kpms.init_model(data, pca=pca, seed=base_seed, **config)

        # Pre-fit 10 AR iterations
        import copy
        current_model = copy.deepcopy(base_model)
        for it in range(10):
            current_model = keypoint_slds.resample_model(
                data, **current_model, ar_only=True, jitter=1e-3, parallel_message_passing=False
            )
        
        init_stats = compute_syllable_stats(current_model["states"]["z"])
        print(f"  Base AR State (Iter 10): Median={init_stats['median']}f ({init_stats['median']/0.06:.1f}ms) | "
              f"Mean={init_stats['mean']:.1f}f | Sylls={init_stats['active']} | Bouts={init_stats['n_bouts']}")

        # 4. Define Test Arms
        # We will run 15 SLDS iterations on each arm and track syllable stats at every iteration
        arms = {
            "Arm 1: 100% Unpatched Upstream JAX": {
                "apply_patch": False,
                "override": {}
            },
            "Arm 2: 100% Monkeypatched Production": {
                "apply_patch": True,
                "override": {"use_banded_continuous": True, "use_numba_hmm": True}
            },
            "Arm 3: Monkeypatch EXCEPT Kalman (Unpatched x)": {
                "apply_patch": True,
                "override": {"use_banded_continuous": False, "use_numba_hmm": True}
            },
            "Arm 4: Monkeypatch EXCEPT FFBS (Unpatched z)": {
                "apply_patch": True,
                "override": {"use_banded_continuous": True, "use_numba_hmm": False}
            }
        }

        NUM_SLDS_ITERS = 15
        results = {}

        print(f"\n[Step 2/3] Running {NUM_SLDS_ITERS} SLDS iterations across {len(arms)} arms...")

        for arm_name, arm_cfg in arms.items():
            print(f"\n>>> Running {arm_name}...")
            rb.restore_original_kernels()
            if arm_cfg["apply_patch"]:
                rb.apply_optimizations()
                for k, v in arm_cfg["override"].items():
                    setattr(rb.CONFIG, k, v)
                # Re-apply setting
                if "use_banded_continuous" in arm_cfg["override"]:
                    if arm_cfg["override"]["use_banded_continuous"]:
                        slds_gibbs.resample_continuous_stateseqs = rb.pure_numba_resample_continuous_stateseqs
                        keypoint_slds.resample_continuous_stateseqs = rb.pure_numba_resample_continuous_stateseqs
                    else:
                        slds_gibbs.resample_continuous_stateseqs = rb._ORIGINAL_FUNCTIONS["kp_slds_resample_x"]
                        keypoint_slds.resample_continuous_stateseqs = rb._ORIGINAL_FUNCTIONS["kp_slds_resample_x"]

            rb.KPMSStateCache.clear()
            arm_model = copy.deepcopy(current_model)
            # Use identical random seed progression
            arm_model["seed"] = jr.PRNGKey(999)

            arm_history = []
            t_arm_start = time.time()

            for slds_it in range(1, NUM_SLDS_ITERS + 1):
                arm_model = keypoint_slds.resample_model(
                    data, **arm_model, ar_only=False, jitter=1e-3, parallel_message_passing=False
                )
                stats = compute_syllable_stats(arm_model["states"]["z"])
                stats["iteration"] = slds_it
                arm_history.append(stats)
                print(f"  Iter {slds_it:2d}: Median={stats['median']:4.1f}f ({stats['median']/0.06:5.1f}ms) | "
                      f"Mean={stats['mean']:5.1f}f | p90={stats['p90']:5.1f}f | Sylls={stats['active']:2d} | Bouts={stats['n_bouts']:3d}")

            t_arm_elapsed = time.time() - t_arm_start
            print(f"  --> {arm_name} finished in {t_arm_elapsed:.2f}s ({t_arm_elapsed/NUM_SLDS_ITERS:.2f}s/it)")
            results[arm_name] = {
                "history": arm_history,
                "elapsed": t_arm_elapsed
            }

        # 5. Clean up kernels
        rb.restore_original_kernels()

        # 6. Comparative Synthesis Report
        print("\n" + "=" * 90)
        print("                         COMPARATIVE MULTI-ARM AUDIT RESULTS")
        print("=" * 90)
        print(f"{'Iteration':10s} | {'Unpatched JAX (Ref)':22s} | {'Full Monkeypatched':22s} | {'Unpatched Kalman (x)':22s}")
        print(f"{'':10s} | {'Median / Mean / Bouts':22s} | {'Median / Mean / Bouts':22s} | {'Median / Mean / Bouts':22s}")
        print("-" * 90)

        unpatched_hist = results["Arm 1: 100% Unpatched Upstream JAX"]["history"]
        patched_hist = results["Arm 2: 100% Monkeypatched Production"]["history"]
        unpatch_x_hist = results["Arm 3: Monkeypatch EXCEPT Kalman (Unpatched x)"]["history"]

        for it_idx in range(NUM_SLDS_ITERS):
            u = unpatched_hist[it_idx]
            p = patched_hist[it_idx]
            ux = unpatch_x_hist[it_idx]
            it_num = u["iteration"]
            u_str = f"{u['median']:4.1f}f / {u['mean']:4.1f}f / {u['n_bouts']:3d}"
            p_str = f"{p['median']:4.1f}f / {p['mean']:4.1f}f / {p['n_bouts']:3d}"
            ux_str = f"{ux['median']:4.1f}f / {ux['mean']:4.1f}f / {ux['n_bouts']:3d}"
            print(f"SLDS Iter {it_num:2d} | {u_str:22s} | {p_str:22s} | {ux_str:22s}")

        print("=" * 90)
        print("\nDuration Progression Trends:")
        print(f"  * Unpatched JAX:        Iter 1 Median={unpatched_hist[0]['median']}f -> Iter {NUM_SLDS_ITERS} Median={unpatched_hist[-1]['median']}f (Mean: {unpatched_hist[0]['mean']:.1f}f -> {unpatched_hist[-1]['mean']:.1f}f)")
        print(f"  * Full Monkeypatch:     Iter 1 Median={patched_hist[0]['median']}f -> Iter {NUM_SLDS_ITERS} Median={patched_hist[-1]['median']}f (Mean: {patched_hist[0]['mean']:.1f}f -> {patched_hist[-1]['mean']:.1f}f)")
        print(f"  * Unpatched Kalman (x): Iter 1 Median={unpatch_x_hist[0]['median']}f -> Iter {NUM_SLDS_ITERS} Median={unpatch_x_hist[-1]['median']}f (Mean: {unpatch_x_hist[0]['mean']:.1f}f -> {unpatch_x_hist[-1]['mean']:.1f}f)")


if __name__ == "__main__":
    from jax_moseq.models.keypoint_slds import gibbs as slds_gibbs
    run_experiment()
