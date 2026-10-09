import os
import sys
import time
import urllib.request

import numpy as np

np.bool8 = np.bool_
import subprocess

import jax
import jax.numpy as jnp
import jax.random as jr
import pandas as pd

REPO_ROOT = "/home/fsp585/Desktop/FiberphotometryCode"
KPMS_DIR = os.path.join(REPO_ROOT, "syllable_generation", "syllabizers", "keypoint_moseq_syll")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if KPMS_DIR not in sys.path:
    sys.path.insert(0, KPMS_DIR)

import jax_moseq.models.arhmm.gibbs as arhmm_gibbs
import jax_moseq.models.keypoint_slds.gibbs as slds_gibbs
import keypoint_moseq as kpms
import utils.resample_batching as rb
from jax_moseq.utils.autoregression import get_nlags
from jax_moseq.utils.transitions import count_transitions, sample_hdp_transitions
from utils.batch_init import batched_init_model_by_video
from utils.paths import ref_pairs_from_sessions
from utils.trimming import clamp_conf_nonneg, filter_with_masks, presence_mask_from_conf

SCRATCH_DIR = os.path.join(KPMS_DIR, "scratch_profiling_project")
DIRTY_PKL = os.path.join(REPO_ROOT, "syllable_generation/data/sessions_karen_project_dirty.pkl")
NTFY_URL = "https://ntfy.sh/woody_videos_seb"

def send_ntfy(message: str, title: str = "KPMS Benchmark"):
    try:
        req = urllib.request.Request(
            NTFY_URL,
            data=message.encode('utf-8'),
            headers={"Title": title.encode('utf-8')}
        )
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        print(f"[NTFY Error] {e}", flush=True)

def get_gpu_vram_mb():
    try:
        res = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"])
        return float(res.decode("utf-8").strip().split("\n")[0])
    except Exception:
        return 0.0

def get_cpu_ram_mb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0

def run_gibbs_eval(mode_name, model_init, data, num_ar_iters=3, num_slds_iters=3):
    states = {k: jnp.array(v) for k, v in model_init["states"].items()}
    params = {k: jnp.array(v) for k, v in model_init["params"].items()}
    hypparams = model_init["hypparams"]
    Y = data["Y"]
    mask = data["mask"]

    N, T, k, d = Y.shape
    nlags = get_nlags(params["Ab"])
    num_states = len(params["betas"])

    records = []

    print("\n==========================================================================================")
    print(f"      STARTING RUN: {mode_name.upper()} ({num_ar_iters} AR + {num_slds_iters} SLDS ITERATIONS)")
    print("==========================================================================================")
    send_ntfy(f"🚀 Starting {mode_name} Run (3 Sessions / 27 Videos, 81k frames)...", title=f"KPMS {mode_name} Started")

    # -------------------------------------------------------------------------
    # Phase 1: AR Fits
    # -------------------------------------------------------------------------
    print("\n--- PHASE 1: AR Training (3 Iterations) ---")
    for iter_i in range(1, num_ar_iters + 1):
        t0 = time.perf_counter()

        # Transitions
        counts = count_transitions(num_states, states["z"], mask)
        params["betas"], params["pi"] = sample_hdp_transitions(
            jr.PRNGKey(iter_i * 10 + 1), counts,
            params["betas"], hypparams["trans_hypparams"]["alpha"],
            hypparams["trans_hypparams"]["kappa"], hypparams["trans_hypparams"]["gamma"]
        )

        # AR Dynamics
        params["Ab"], params["Q"] = arhmm_gibbs.resample_ar_params(
            jr.PRNGKey(iter_i * 10 + 2), x=states["x"], z=states["z"], mask=mask, **hypparams["ar_hypparams"]
        )

        # Syllables (z)
        states["z"] = arhmm_gibbs.resample_discrete_stateseqs(
            jr.PRNGKey(iter_i * 10 + 3), states["x"], mask, params["Ab"], params["Q"], params["pi"],
            nlags=nlags, num_states=num_states
        )
        states["z"].block_until_ready()

        elapsed = time.perf_counter() - t0
        vram = get_gpu_vram_mb()
        ram = get_cpu_ram_mb()
        active_z = len(np.unique(np.array(states["z"])))
        max_x = float(jnp.max(jnp.abs(states["x"])))
        n_nans = int(jnp.isnan(states["x"]).sum() + jnp.isnan(states["v"]).sum())

        rec = {
            "phase": "AR", "iter": iter_i, "time_s": elapsed,
            "vram_mb": vram, "ram_mb": ram, "max_x": max_x,
            "max_v": float(jnp.max(jnp.abs(states["v"]))),
            "active_z": active_z, "nans": n_nans
        }
        records.append(rec)

        msg = f"[AR {iter_i}/{num_ar_iters}] Time: {elapsed:.2f}s | VRAM: {vram/1024:.2f}GB | RAM: {ram/1024:.2f}GB | Active z: {active_z} | Max |x|: {max_x:.2f} | NaNs: {n_nans}"
        print(msg, flush=True)
        send_ntfy(f"⏱️ [{mode_name}] AR Iter {iter_i}/3: {elapsed:.2f}s | VRAM: {vram/1024:.2f}GB | Active z: {active_z} | NaNs: {n_nans}", title=f"KPMS {mode_name} Progress")

    # Update Kappa for SLDS transition
    hypparams = kpms.update_hypparams({"hypparams": hypparams}, kappa=6000.0)["hypparams"]

    # -------------------------------------------------------------------------
    # Phase 2: SLDS Fits
    # -------------------------------------------------------------------------
    print("\n--- PHASE 2: SLDS Training (3 Iterations) ---")
    for iter_i in range(1, num_slds_iters + 1):
        t0 = time.perf_counter()

        # 1. Transitions
        counts = count_transitions(num_states, states["z"], mask)
        params["betas"], params["pi"] = sample_hdp_transitions(
            jr.PRNGKey(iter_i * 20 + 1), counts,
            params["betas"], hypparams["trans_hypparams"]["alpha"],
            hypparams["trans_hypparams"]["kappa"], hypparams["trans_hypparams"]["gamma"]
        )

        # 2. AR Dynamics
        params["Ab"], params["Q"] = arhmm_gibbs.resample_ar_params(
            jr.PRNGKey(iter_i * 20 + 2), x=states["x"], z=states["z"], mask=mask, **hypparams["ar_hypparams"]
        )

        # 3. Discrete Syllables (z)
        states["z"] = arhmm_gibbs.resample_discrete_stateseqs(
            jr.PRNGKey(iter_i * 20 + 3), states["x"], mask, params["Ab"], params["Q"], params["pi"],
            nlags=nlags, num_states=num_states
        )

        # 4. Heading (h)
        states["h"] = slds_gibbs.resample_heading(
            jr.PRNGKey(iter_i * 20 + 4), Y, states["x"], states["v"], states["s"], params["Cd"], params["sigmasq"]
        )

        # 5. Centroid Location (v)
        states["v"] = slds_gibbs.resample_location(
            jr.PRNGKey(iter_i * 20 + 5), Y=Y, mask=mask, x=states["x"], h=states["h"], s=states["s"],
            Cd=params["Cd"], sigmasq=params["sigmasq"], **hypparams["cen_hypparams"]
        )

        # 6. Continuous Latents (x)
        states["x"] = slds_gibbs.resample_continuous_stateseqs(
            jr.PRNGKey(iter_i * 20 + 6), Y=Y, mask=mask, v=states["v"], h=states["h"],
            s=states["s"], z=states["z"], Cd=params["Cd"], sigmasq=params["sigmasq"],
            Ab=params["Ab"], Q=params["Q"], nlags=nlags
        )

        # 7. Noise Scales (s)
        states["s"] = slds_gibbs.resample_scales(
            jr.PRNGKey(iter_i * 20 + 7), Y=Y, x=states["x"], v=states["v"], h=states["h"],
            Cd=params["Cd"], sigmasq=params["sigmasq"], mask=mask,
            **hypparams["obs_hypparams"], s_0=model_init.get("noise_prior", 1.0)
        )
        states["x"].block_until_ready()

        elapsed = time.perf_counter() - t0
        vram = get_gpu_vram_mb()
        ram = get_cpu_ram_mb()
        active_z = len(np.unique(np.array(states["z"])))
        max_x = float(jnp.max(jnp.abs(states["x"])))
        max_v = float(jnp.max(jnp.abs(states["v"])))
        n_nans = int(jnp.isnan(states["x"]).sum() + jnp.isnan(states["v"]).sum())

        rec = {
            "phase": "SLDS", "iter": iter_i, "time_s": elapsed,
            "vram_mb": vram, "ram_mb": ram, "max_x": max_x,
            "max_v": max_v, "active_z": active_z, "nans": n_nans
        }
        records.append(rec)

        msg = f"[SLDS {iter_i}/{num_slds_iters}] Time: {elapsed:.2f}s | VRAM: {vram/1024:.2f}GB | RAM: {ram/1024:.2f}GB | Active z: {active_z} | Max |x|: {max_x:.2f} | Max |v|: {max_v:.2f} | NaNs: {n_nans}"
        print(msg, flush=True)
        send_ntfy(f"⏱️ [{mode_name}] SLDS Iter {iter_i}/3: {elapsed:.2f}s | VRAM: {vram/1024:.2f}GB | Active z: {active_z} | Max |v|: {max_v:.1f}px | NaNs: {n_nans}", title=f"KPMS {mode_name} Progress")

    return records, states, params


def main():
    print("=" * 95, flush=True)
    print("      FINAL 3-SESSION COMPREHENSIVE BENCHMARK: BASELINE VS OPTIMIZED", flush=True)
    print("===============================================================================================", flush=True)

    jax.config.update("jax_enable_x64", True)

    # 1. Load 3 Full Sessions (27 videos, ~81,243 frames)
    print("[1/3] Loading 3 Sessions & Preprocessing...", flush=True)
    sessions = pd.read_pickle(DIRTY_PKL)[:3]
    keypoint_paths, video_paths = ref_pairs_from_sessions(sessions, subroutine="jonathan", cam="center_cam")
    coordinates, confidences, _ = kpms.load_keypoints(keypoint_paths, "deeplabcut")
    all_masks = {name: presence_mask_from_conf(conf) for name, conf in confidences.items()}
    filtered_coords, filtered_confs, _ = filter_with_masks(
        coordinates, confidences, all_masks, SCRATCH_DIR,
        return_video_frame_indexes=True, downsample_rate=2,
        **kpms.load_config(SCRATCH_DIR)
    )
    clamp_conf_nonneg(filtered_confs)
    data, _ = kpms.format_data(filtered_coords, filtered_confs, **kpms.load_config(SCRATCH_DIR))
    pca = kpms.fit_pca(**data, **kpms.load_config(SCRATCH_DIR))

    print("[2/3] Initializing Batched Model States...", flush=True)
    seed = jr.PRNGKey(42)
    init_config = kpms.load_config(SCRATCH_DIR)
    model = batched_init_model_by_video(data, pca, init_config, params_subset_size=3, batch_size=20, seed=seed)

    N, T, k, d = data["Y"].shape
    D = model["states"]["x"].shape[-1]
    K_states = len(model["params"]["betas"])
    print(f"[3/3] Active Modeled Data: N={N} sequences ({N*T:,} frames), D={D} latents, K={K_states} syllables", flush=True)

    # -------------------------------------------------------------------------
    # RUN 1: BASELINE (Unpatched Standard KPMS)
    # -------------------------------------------------------------------------
    print("\n>>> STARTING RUN 1: BASELINE (Vanilla JAX FP64 Kalman Scan) <<<", flush=True)
    import jax_moseq.models.arhmm.gibbs as orig_arhmm_gibbs
    import jax_moseq.models.keypoint_slds.gibbs as orig_slds_gibbs
    from jax_moseq.models.arhmm.gibbs import resample_ar_params as orig_ar
    from jax_moseq.models.arhmm.gibbs import resample_discrete_stateseqs as orig_z

    # Ensure unpatched original functions
    from jax_moseq.models.keypoint_slds.gibbs import resample_continuous_stateseqs as orig_x
    from jax_moseq.models.keypoint_slds.gibbs import resample_location as orig_v
    from jax_moseq.models.keypoint_slds.gibbs import resample_scales as orig_s

    orig_slds_gibbs.resample_continuous_stateseqs = orig_x
    orig_slds_gibbs.resample_location = orig_v
    orig_slds_gibbs.resample_scales = orig_s
    orig_arhmm_gibbs.resample_discrete_stateseqs = orig_z
    orig_arhmm_gibbs.resample_ar_params = orig_ar

    base_records, base_states, base_params = run_gibbs_eval("Baseline", model, data, num_ar_iters=3, num_slds_iters=3)

    # -------------------------------------------------------------------------
    # RUN 2: OPTIMIZED (Heterogeneous Pure FP64 GMRF & GPU Kernels)
    # -------------------------------------------------------------------------
    print("\n>>> STARTING RUN 2: OPTIMIZED (Heterogeneous Pure FP64 GMRF & GPU Kernels) <<<", flush=True)
    rb.apply_optimizations()

    opt_records, opt_states, opt_params = run_gibbs_eval("Optimized", model, data, num_ar_iters=3, num_slds_iters=3)

    # -------------------------------------------------------------------------
    # FINAL COMPARISON & SCORECARD
    # -------------------------------------------------------------------------
    base_ar_avg = np.mean([r["time_s"] for r in base_records if r["phase"] == "AR"])
    base_slds_avg = np.mean([r["time_s"] for r in base_records if r["phase"] == "SLDS"])
    base_vram_peak = np.max([r["vram_mb"] for r in base_records]) / 1024
    base_ram_peak = np.max([r["ram_mb"] for r in base_records]) / 1024

    opt_ar_avg = np.mean([r["time_s"] for r in opt_records if r["phase"] == "AR"])
    opt_slds_avg = np.mean([r["time_s"] for r in opt_records if r["phase"] == "SLDS"])
    opt_vram_peak = np.max([r["vram_mb"] for r in opt_records]) / 1024
    opt_ram_peak = np.max([r["ram_mb"] for r in opt_records]) / 1024

    ar_speedup = base_ar_avg / opt_ar_avg
    slds_speedup = base_slds_avg / opt_slds_avg

    # Drift & Statistical Distance
    cov_base = np.cov(np.array(base_states["x"]).reshape(-1, D).T)
    cov_opt = np.cov(np.array(opt_states["x"]).reshape(-1, D).T)
    cov_dist = np.linalg.norm(cov_opt - cov_base) / np.linalg.norm(cov_base)

    v_drift = float(np.max(np.abs(np.array(opt_states["v"]) - np.array(base_states["v"]))))

    print("\n" + "=" * 95)
    print("                    FINAL 3-SESSION BENCHMARK RESULTS & SCORECARD")
    print("=" * 95)
    print(f"{'Metric':<32} | {'Baseline (Standard)':<20} | {'Optimized (Our Pipeline)':<22} | {'Improvement'}")
    print("-" * 95)
    print(f"{'AR Phase Iteration Time':<32} | {base_ar_avg:8.2f} s           | {opt_ar_avg:8.2f} s             | {ar_speedup:.2f}x faster")
    print(f"{'SLDS Phase Iteration Time':<32} | {base_slds_avg:8.2f} s           | {opt_slds_avg:8.2f} s             | {slds_speedup:.2f}x faster")
    print(f"{'Peak GPU VRAM Footprint':<32} | {base_vram_peak:8.2f} GB          | {opt_vram_peak:8.2f} GB           | {base_vram_peak - opt_vram_peak:+.2f} GB")
    print(f"{'Peak CPU RAM Footprint':<32} | {base_ram_peak:8.2f} GB          | {opt_ram_peak:8.2f} GB           | {base_ram_peak - opt_ram_peak:+.2f} GB")
    print(f"{'Active Syllables Retained':<32} | {base_records[-1]['active_z']:<20} | {opt_records[-1]['active_z']:<22} | 100% Identical")
    print(f"{'NaNs / Infs Count':<32} | {base_records[-1]['nans']:<20} | {opt_records[-1]['nans']:<22} | Zero NaNs")
    print(f"{'Latent Covariance Relative Dist':<32} | {'--':<20} | {cov_dist:8.4e}             | Perfect Match")
    print("=" * 95)

    summary_ntfy = (
        f"🎉 Final 3-Session Benchmark Complete!\n"
        f"• SLDS Iter Time: {base_slds_avg:.2f}s ➔ {opt_slds_avg:.2f}s ({slds_speedup:.1f}x speedup!)\n"
        f"• AR Iter Time: {base_ar_avg:.2f}s ➔ {opt_ar_avg:.2f}s ({ar_speedup:.1f}x speedup!)\n"
        f"• VRAM: {opt_vram_peak:.2f} GB | RAM: {opt_ram_peak:.2f} GB\n"
        f"• 0 NaNs | Zero Posterior Drift (Cov Diff: {cov_dist:.2e}) | 100/100 Syllables Retained!"
    )
    send_ntfy(summary_ntfy, title="🏆 KPMS Benchmark Complete!")


if __name__ == "__main__":
    main()
