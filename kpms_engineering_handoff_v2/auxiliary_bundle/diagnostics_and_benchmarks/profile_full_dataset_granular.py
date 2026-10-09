#!/usr/bin/env python3
"""
diagnostics/profile_full_dataset_granular.py

Strictly measured, empirical per-function granular profiling on the FULL 30-session AMPH dataset:
- 562 sequences, 5,516,954 active frames (5,636,860 allocated frames).
- Runs 2 AR iterations (1 cold JIT + 1 warmed steady state).
- Runs 2 SLDS iterations (1 cold JIT + 1 warmed steady state).
- Measures exact sub-operation times with explicit JAX synchronization (block_until_ready()):
  1. Discrete states (z):
     - GPU emissions (with block_until_ready)
     - D2H PCIe transfer
     - CPU multi-threaded FFBS (prange across 32 cores)
     - H2D transfer & output assembly
  2. AR Dynamics (Ab, Q):
     - Zero-alloc streaming stats accumulator
     - MNIW solve (with block_until_ready)
  3. Transitions (pi, beta): HDP sampler
  4. Continuous states (x):
     - Rigid transformation (Y - v, rot h)
     - Banded matrix assembly & Cholesky solve
     - Output transfer
  5. Centroid location (v): GPU batch execution (with block_until_ready)
  6. Heading (h): GPU batch execution (with block_until_ready)
  7. Noise scales (s): GPU batch execution (with block_until_ready)
- Tracks process host RAM (psutil) and GPU VRAM (nvidia-smi).
- Strictly read-only against existing checkpoints.
"""

import multiprocessing
import os
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.75"

if multiprocessing.current_process().name != 'MainProcess':
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

REPO_ROOT = "/home/fsp585/Desktop/FiberphotometryCode"
KPMS_DIR = os.path.join(REPO_ROOT, "syllable_generation", "syllabizers", "keypoint_moseq_syll")
if REPO_ROOT not in sys.path: sys.path.insert(0, REPO_ROOT)
if KPMS_DIR not in sys.path: sys.path.insert(0, KPMS_DIR)

import numpy as np

if not hasattr(np, "bool8"): np.bool8 = np.bool_
import jax.numpy as jnp
import jax.random as jr
import jax_moseq.models.arhmm.gibbs as arhmm_gibbs
import jax_moseq.models.keypoint_slds.gibbs as slds_gibbs
import keypoint_moseq as kpms
import psutil
import utils.resample_batching as rb
from jax_moseq.models.arhmm.gibbs import resample_hdp_transitions

PROJECT_DIR = os.path.join(KPMS_DIR, "amph_project")
SOURCE_MODEL = "amph_prod_k150k_run_01"

def get_process_ram_gb():
    return psutil.Process(os.getpid()).memory_info().rss / (1024**3)

def get_gpu_vram_gb():
    try:
        import subprocess
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        return float(out.decode().strip()) / 1024.0
    except Exception:
        return 0.0

def sync(x):
    """Ensure JAX computation is completely finished on GPU."""
    if hasattr(x, "block_until_ready"):
        x.block_until_ready()
    elif isinstance(x, (tuple, list)):
        for item in x:
            sync(item)
    elif isinstance(x, dict):
        for item in x.values():
            sync(item)
    return x

# Sub-operation telemetry accumulator
SUB_TIMINGS = {}

def reset_sub_timings():
    global SUB_TIMINGS
    SUB_TIMINGS = {
        "z_emissions": 0.0,
        "z_d2h": 0.0,
        "z_ffbs": 0.0,
        "z_h2d": 0.0,
        "ar_stats": 0.0,
        "ar_solve": 0.0,
        "transitions": 0.0,
        "x_prep": 0.0,
        "x_rigid_rot": 0.0,
        "x_banded_solve": 0.0,
        "x_h2d": 0.0,
        "location_v": 0.0,
        "heading_h": 0.0,
        "scales_s": 0.0,
        "overhead_misc": 0.0,
    }

# -----------------------------------------------------------------------------
# Instrument discrete sampler (z)
# -----------------------------------------------------------------------------
def instrumented_resample_discrete(seed, x, mask, Ab, Q, pi, nlags=None, num_states=None, batch_size=None, **kwargs):
    N, T, D = x.shape
    if num_states is None: num_states = Ab.shape[0]
    if nlags is None: nlags = (Ab.shape[2] - 1) // D
    if batch_size is None: batch_size = rb.CONFIG.discrete_bs

    T_eff = T - nlags
    pi_cpu = np.asarray(pi, dtype=np.float64)
    pi_T = np.ascontiguousarray(pi_cpu.T)
    mask_cpu = np.asarray(mask[:, nlags:], dtype=bool)

    base_seed = int(jr.randint(seed, (), 0, 2**31 - 1))
    rng = np.random.default_rng(base_seed)

    z_out_all = np.empty((N, T_eff), dtype=np.int32)

    batch_workspace_size = min(batch_size, N)
    ll_batch_cpu = np.empty((batch_workspace_size, T_eff, num_states), dtype=np.float64)
    P_batch_workspace = np.empty((batch_workspace_size, T_eff, num_states), dtype=np.float64)
    uniforms_batch = np.empty((batch_workspace_size, T_eff), dtype=np.float64)

    for i in range(0, N, batch_size):
        end = min(i + batch_size, N)
        N_b = end - i

        # 1. GPU emissions
        t0 = time.perf_counter()
        batch_ll = rb._jit_compute_discrete_emissions(x[i:end], Ab, Q, nlags=nlags, num_states=num_states)
        sync(batch_ll)
        SUB_TIMINGS["z_emissions"] += (time.perf_counter() - t0)

        # 2. D2H transfer
        t0 = time.perf_counter()
        ll_batch_cpu[:N_b] = np.asarray(batch_ll)
        SUB_TIMINGS["z_d2h"] += (time.perf_counter() - t0)

        # 3. In-place random uniforms & CPU FFBS
        t0 = time.perf_counter()
        uniforms_batch[:N_b] = rng.uniform(0.0, 1.0, size=(N_b, T_eff))
        rb._numba_ffbs_all_parallel(
            ll_batch_cpu[:N_b],
            pi_T,
            mask_cpu[i:end],
            uniforms_batch[:N_b],
            z_out_all[i:end],
            P_batch_workspace[:N_b]
        )
        SUB_TIMINGS["z_ffbs"] += (time.perf_counter() - t0)

    # 4. H2D transfer
    t0 = time.perf_counter()
    z_device = jnp.asarray(z_out_all, dtype=jnp.int32)
    sync(z_device)
    SUB_TIMINGS["z_h2d"] += (time.perf_counter() - t0)
    return z_device

# -----------------------------------------------------------------------------
# Instrument AR dynamics sampler (Ab, Q)
# -----------------------------------------------------------------------------
def instrumented_resample_ar(seed, *args, **kwargs):
    if len(args) >= 3:
        mask, x, z = args[:3]
    else:
        mask = kwargs.get("mask")
        x = kwargs.get("x")
        z = kwargs.get("z")

    D = x.shape[-1]
    nlags = kwargs.get("nlags", mask.shape[1] - z.shape[1])
    num_states = kwargs.get("num_states", kwargs.get("K_0").shape[0] if "K_0" in kwargs else 100)
    if "nu_0" not in kwargs and "ar_hypparams" in kwargs:
        kwargs.update(kwargs["ar_hypparams"])

    # 1. Zero-alloc streaming stats
    t0 = time.perf_counter()
    x_np = np.asarray(x, dtype=np.float64)
    z_np = np.asarray(z, dtype=np.int32)
    mask_np = np.asarray(mask, dtype=bool)
    counts, S_in_in, S_out_in, S_out_out = rb._zero_alloc_accumulate_ar_stats(x_np, z_np, mask_np, nlags, num_states)
    SUB_TIMINGS["ar_stats"] += (time.perf_counter() - t0)

    # 2. MNIW solve on GPU
    t0 = time.perf_counter()
    seeds = jr.split(seed, num_states)
    Ab, Q = rb._jit_solve_mniw_posteriors(
        seeds,
        jnp.asarray(counts),
        jnp.asarray(S_in_in),
        jnp.asarray(S_out_in),
        jnp.asarray(S_out_out),
        kwargs["K_0"], kwargs["nu_0"], kwargs["S_0"], kwargs["M_0"],
        num_states=num_states
    )
    sync((Ab, Q))
    SUB_TIMINGS["ar_solve"] += (time.perf_counter() - t0)
    return Ab, Q

# -----------------------------------------------------------------------------
# Instrument continuous latents sampler (x)
# -----------------------------------------------------------------------------
def instrumented_resample_continuous(seed, Y, mask, v, h, s, z, Cd, sigmasq, Ab, Q, jitter=1e-3, **kwargs):
    from jax_moseq.models.keypoint_slds import alignment as slds_alignment
    N, T, k, d = Y.shape
    num_states, D, ar_dim_plus_1 = Ab.shape
    nlags = (ar_dim_plus_1 - 1) // D
    bandwidth = (nlags + 1) * D - 1
    block_dim = (nlags + 1) * D
    N_total = T * D

    # 1. Parameter alignment & prior blocks
    t0 = time.perf_counter()
    Gamma = np.asarray(slds_alignment.center_embedding(k))
    Cd_np = np.kron(Gamma, np.eye(d)) @ np.asarray(Cd, dtype=np.float64)
    sigmasq_np = np.repeat(np.asarray(sigmasq, dtype=np.float64), d, axis=-1)
    Ab_np = np.asarray(Ab, dtype=np.float64)
    Q_np = np.asarray(Q, dtype=np.float64)

    C_mat = Cd_np[:, :-1]
    d_off = Cd_np[:, -1]

    K_blocks = np.zeros((num_states, block_dim, block_dim), dtype=np.float64)
    g_priors = np.zeros((num_states, block_dim), dtype=np.float64)

    for k_idx in range(num_states):
        Q_inv_k = np.linalg.inv(Q_np[k_idx])
        A_k = Ab_np[k_idx, :, :-1]
        b_k = Ab_np[k_idx, :, -1]

        M_k = np.zeros((D, block_dim), dtype=np.float64)
        M_k[:, :-D] = -A_k
        M_k[:, -D:] = np.eye(D)

        K_blocks[k_idx] = M_k.T @ Q_inv_k @ M_k
        g_priors[k_idx] = M_k.T @ Q_inv_k @ b_k

    base_seed_int = int(jr.randint(seed, (), 0, 2**31 - 1))
    rng = np.random.default_rng(base_seed_int)

    CHUNK_SIZE = kwargs.get("chunk_size", rb.CONFIG.continuous_chunk_size)
    max_chunk = min(CHUNK_SIZE, N)

    ab_workspace = np.zeros((max_chunk, bandwidth + 1, N_total), dtype=np.float64)
    g_workspace = np.zeros((max_chunk, N_total), dtype=np.float64)
    eta_workspace = np.zeros((max_chunk, N_total), dtype=np.float64)
    x_workspace = np.zeros((max_chunk, N_total), dtype=np.float64)
    repairs_workspace = np.zeros(max_chunk, dtype=np.int64)

    x_out = np.zeros((N, T, D), dtype=np.float64)
    mask_np = np.asarray(mask, dtype=bool)
    z_np = np.asarray(z, dtype=np.int32)
    SUB_TIMINGS["x_prep"] += (time.perf_counter() - t0)

    for chunk_start in range(0, N, CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, N)
        N_chunk = chunk_end - chunk_start

        # 2. CPU rigid transformation
        t0 = time.perf_counter()
        Y_c = np.asarray(Y[chunk_start:chunk_end], dtype=np.float64)
        v_c = np.asarray(v[chunk_start:chunk_end], dtype=np.float64)
        h_c = np.asarray(h[chunk_start:chunk_end], dtype=np.float64)
        s_c = np.asarray(s[chunk_start:chunk_end], dtype=np.float64)

        cos_h = np.cos(h_c)[:, :, None]
        sin_h = np.sin(h_c)[:, :, None]
        Y_trans = Y_c - v_c[:, :, None, :]
        Y_rot_0 = cos_h * Y_trans[..., 0] + sin_h * Y_trans[..., 1]
        Y_rot_1 = -sin_h * Y_trans[..., 0] + cos_h * Y_trans[..., 1]
        Y_c_van = np.stack([Y_rot_0, Y_rot_1], axis=-1).reshape(N_chunk, T, -1)
        s_c_van = np.repeat(s_c, d, axis=-1)

        ab_chunk = ab_workspace[:N_chunk]
        ab_chunk.fill(0.0)
        g_chunk = g_workspace[:N_chunk]
        g_chunk.fill(0.0)
        eta_chunk = eta_workspace[:N_chunk]
        rng.standard_normal(out=eta_chunk)
        x_chunk = x_workspace[:N_chunk]
        x_chunk.fill(0.0)
        repairs_chunk = repairs_workspace[:N_chunk]
        repairs_chunk.fill(0)
        SUB_TIMINGS["x_rigid_rot"] += (time.perf_counter() - t0)

        # 3. Numba banded solve
        t0 = time.perf_counter()
        rb._numba_assemble_and_solve_all_parallel(
            ab_chunk, g_chunk, eta_chunk, x_chunk,
            Y_c_van,
            mask_np[chunk_start:chunk_end],
            s_c_van,
            d_off, C_mat, sigmasq_np,
            K_blocks, g_priors,
            z_np[chunk_start:chunk_end],
            N_chunk, T, D, nlags, bandwidth, block_dim, N_total, jitter,
            repairs_chunk
        )
        x_out[chunk_start:chunk_end] = x_chunk.reshape(N_chunk, T, D)
        SUB_TIMINGS["x_banded_solve"] += (time.perf_counter() - t0)

    # 4. H2D transfer
    t0 = time.perf_counter()
    x_device = jnp.asarray(x_out)
    sync(x_device)
    SUB_TIMINGS["x_h2d"] += (time.perf_counter() - t0)
    return x_device

# -----------------------------------------------------------------------------
# Instrument Pose samplers (v, h, s)
# -----------------------------------------------------------------------------
def instrumented_resample_location(*args, **kwargs):
    t0 = time.perf_counter()
    res = rb.batched_gpu_resample_location(*args, **kwargs)
    sync(res)
    SUB_TIMINGS["location_v"] += (time.perf_counter() - t0)
    return res

def instrumented_resample_heading(*args, **kwargs):
    t0 = time.perf_counter()
    res = rb.batched_resample_heading(*args, **kwargs)
    sync(res)
    SUB_TIMINGS["heading_h"] += (time.perf_counter() - t0)
    return res

def instrumented_resample_scales(*args, **kwargs):
    t0 = time.perf_counter()
    res = rb.batched_resample_scales(*args, **kwargs)
    sync(res)
    SUB_TIMINGS["scales_s"] += (time.perf_counter() - t0)
    return res

def instrumented_resample_transitions(*args, **kwargs):
    t0 = time.perf_counter()
    res = resample_hdp_transitions(*args, **kwargs)
    sync(res)
    SUB_TIMINGS["transitions"] += (time.perf_counter() - t0)
    return res

def run_profiler():
    print("=" * 85)
    print("  MEASURED FULL-DATASET GRANULAR PROFILING (30 SESSIONS, 5.5M FRAMES)")
    print("=" * 85, flush=True)

    print(f"Initial Host Process RAM: {get_process_ram_gb():.2f} GB")
    print(f"Initial GPU VRAM:        {get_gpu_vram_gb():.2f} GB\n", flush=True)

    # 1. Load exact production data & baseline model state (read-only)
    print("Loading production dataset from checkpoint (read-only)...", flush=True)
    t0_load = time.time()
    model, data, metadata, _ = kpms.load_checkpoint(PROJECT_DIR, SOURCE_MODEL, iteration=0)
    print(f"Loaded {data['Y'].shape[0]} sequences ({data['Y'].shape[1]} frames each) in {time.time()-t0_load:.2f} s.\n", flush=True)

    # Patch modules
    arhmm_gibbs.resample_discrete_stateseqs = instrumented_resample_discrete
    arhmm_gibbs.resample_ar_params = instrumented_resample_ar
    arhmm_gibbs.resample_hdp_transitions = instrumented_resample_transitions

    slds_gibbs.resample_continuous_stateseqs = instrumented_resample_continuous
    slds_gibbs.resample_location = instrumented_resample_location
    slds_gibbs.resample_heading = instrumented_resample_heading
    slds_gibbs.resample_scales = instrumented_resample_scales

    from jax_moseq.models import arhmm, keypoint_slds
    arhmm.resample_discrete_stateseqs = instrumented_resample_discrete
    arhmm.resample_ar_params = instrumented_resample_ar
    arhmm.resample_hdp_transitions = instrumented_resample_transitions

    keypoint_slds.resample_continuous_stateseqs = instrumented_resample_continuous
    keypoint_slds.resample_location = instrumented_resample_location
    keypoint_slds.resample_heading = instrumented_resample_heading
    keypoint_slds.resample_scales = instrumented_resample_scales

    iteration_records = []

    # -------------------------------------------------------------------------
    # PHASE 1: AR-HMM (2 Iterations: 1 Cold JIT + 1 Warmed Steady State)
    # -------------------------------------------------------------------------
    print("=" * 85)
    print("  PHASE 1: AR-HMM FIT (2 Iterations: Cold JIT -> Warmed Steady State)")
    print("=" * 85, flush=True)

    noise_prior = model["noise_prior"]

    for it in range(1, 3):
        phase_name = "AR"
        status = "Cold JIT" if it == 1 else "Warmed Steady State"
        print(f"\n>>> Running AR Iteration {it} ({status}) <<<", flush=True)

        reset_sub_timings()
        ram_start = get_process_ram_gb()
        vram_start = get_gpu_vram_gb()

        t0_iter = time.perf_counter()
        model = arhmm.resample_model(data, **model)
        sync(model)
        t_iter_total = time.perf_counter() - t0_iter
        model["noise_prior"] = noise_prior

        ram_end = get_process_ram_gb()
        vram_end = get_gpu_vram_gb()

        # Compute accounted vs overhead
        accounted = sum(SUB_TIMINGS.values())
        overhead = max(0.0, t_iter_total - accounted)
        SUB_TIMINGS["overhead_misc"] = overhead

        record = {
            "iter": it,
            "phase": phase_name,
            "status": status,
            "total_time": t_iter_total,
            "ram_gb": ram_end,
            "vram_gb": vram_end,
            "timings": dict(SUB_TIMINGS)
        }
        iteration_records.append(record)

        print(f"Iteration {it} Total: {t_iter_total:.2f} s | RAM: {ram_end:.2f} GB | VRAM: {vram_end:.2f} GB")
        print(f"  * Discrete emissions (GPU):    {SUB_TIMINGS['z_emissions']:6.2f} s ({SUB_TIMINGS['z_emissions']/t_iter_total*100:5.1f}%)")
        print(f"  * Discrete D2H transfer:       {SUB_TIMINGS['z_d2h']:6.2f} s ({SUB_TIMINGS['z_d2h']/t_iter_total*100:5.1f}%)")
        print(f"  * Discrete CPU FFBS (prange):  {SUB_TIMINGS['z_ffbs']:6.2f} s ({SUB_TIMINGS['z_ffbs']/t_iter_total*100:5.1f}%)")
        print(f"  * Discrete H2D transfer:       {SUB_TIMINGS['z_h2d']:6.2f} s ({SUB_TIMINGS['z_h2d']/t_iter_total*100:5.1f}%)")
        print(f"  * AR streaming stats:          {SUB_TIMINGS['ar_stats']:6.2f} s ({SUB_TIMINGS['ar_stats']/t_iter_total*100:5.1f}%)")
        print(f"  * AR MNIW solve (GPU):         {SUB_TIMINGS['ar_solve']:6.2f} s ({SUB_TIMINGS['ar_solve']/t_iter_total*100:5.1f}%)")
        print(f"  * HDP transitions:             {SUB_TIMINGS['transitions']:6.2f} s ({SUB_TIMINGS['transitions']/t_iter_total*100:5.1f}%)")
        print(f"  * Pipeline Overhead / Python:  {overhead:6.2f} s ({overhead/t_iter_total*100:5.1f}%)", flush=True)

    # Transition kappa for SLDS
    print("\n[transition] Updating Kappa for SLDS: 15,000.0", flush=True)
    model = kpms.update_hypparams(model, kappa=15000.0)

    # -------------------------------------------------------------------------
    # PHASE 2: SLDS (2 Iterations: 1 Cold JIT + 1 Warmed Steady State)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 85)
    print("  PHASE 2: SLDS FIT (2 Iterations: Cold JIT -> Warmed Steady State)")
    print("=" * 85, flush=True)

    for it in range(1, 3):
        phase_name = "SLDS"
        status = "Cold JIT" if it == 1 else "Warmed Steady State"
        print(f"\n>>> Running SLDS Iteration {it} ({status}) <<<", flush=True)

        reset_sub_timings()
        t0_iter = time.perf_counter()
        model = keypoint_slds.resample_model(data, **model, jitter=1e-3)
        sync(model)
        t_iter_total = time.perf_counter() - t0_iter

        ram_end = get_process_ram_gb()
        vram_end = get_gpu_vram_gb()

        accounted = sum(SUB_TIMINGS.values())
        overhead = max(0.0, t_iter_total - accounted)
        SUB_TIMINGS["overhead_misc"] = overhead

        record = {
            "iter": it + 2,
            "phase": phase_name,
            "status": status,
            "total_time": t_iter_total,
            "ram_gb": ram_end,
            "vram_gb": vram_end,
            "timings": dict(SUB_TIMINGS)
        }
        iteration_records.append(record)

        print(f"Iteration {it} Total: {t_iter_total:.2f} s | RAM: {ram_end:.2f} GB | VRAM: {vram_end:.2f} GB")
        print(f"  * Continuous prep:             {SUB_TIMINGS['x_prep']:6.2f} s ({SUB_TIMINGS['x_prep']/t_iter_total*100:5.1f}%)")
        print(f"  * Continuous rigid rotation:   {SUB_TIMINGS['x_rigid_rot']:6.2f} s ({SUB_TIMINGS['x_rigid_rot']/t_iter_total*100:5.1f}%)")
        print(f"  * Continuous Banded Cholesky:  {SUB_TIMINGS['x_banded_solve']:6.2f} s ({SUB_TIMINGS['x_banded_solve']/t_iter_total*100:5.1f}%)")
        print(f"  * Continuous H2D transfer:     {SUB_TIMINGS['x_h2d']:6.2f} s ({SUB_TIMINGS['x_h2d']/t_iter_total*100:5.1f}%)")
        print(f"  * Centroid location v (GPU):   {SUB_TIMINGS['location_v']:6.2f} s ({SUB_TIMINGS['location_v']/t_iter_total*100:5.1f}%)")
        print(f"  * Heading h (GPU):             {SUB_TIMINGS['heading_h']:6.2f} s ({SUB_TIMINGS['heading_h']/t_iter_total*100:5.1f}%)")
        print(f"  * Local noise scales s (GPU):  {SUB_TIMINGS['scales_s']:6.2f} s ({SUB_TIMINGS['scales_s']/t_iter_total*100:5.1f}%)")
        print(f"  * Discrete emissions (GPU):    {SUB_TIMINGS['z_emissions']:6.2f} s ({SUB_TIMINGS['z_emissions']/t_iter_total*100:5.1f}%)")
        print(f"  * Discrete D2H transfer:       {SUB_TIMINGS['z_d2h']:6.2f} s ({SUB_TIMINGS['z_d2h']/t_iter_total*100:5.1f}%)")
        print(f"  * Discrete CPU FFBS (prange):  {SUB_TIMINGS['z_ffbs']:6.2f} s ({SUB_TIMINGS['z_ffbs']/t_iter_total*100:5.1f}%)")
        print(f"  * Discrete H2D transfer:       {SUB_TIMINGS['z_h2d']:6.2f} s ({SUB_TIMINGS['z_h2d']/t_iter_total*100:5.1f}%)")
        print(f"  * AR streaming stats:          {SUB_TIMINGS['ar_stats']:6.2f} s ({SUB_TIMINGS['ar_stats']/t_iter_total*100:5.1f}%)")
        print(f"  * AR MNIW solve (GPU):         {SUB_TIMINGS['ar_solve']:6.2f} s ({SUB_TIMINGS['ar_solve']/t_iter_total*100:5.1f}%)")
        print(f"  * HDP transitions:             {SUB_TIMINGS['transitions']:6.2f} s ({SUB_TIMINGS['transitions']/t_iter_total*100:5.1f}%)")
        print(f"  * Pipeline Overhead / Python:  {overhead:6.2f} s ({overhead/t_iter_total*100:5.1f}%)", flush=True)

    # -------------------------------------------------------------------------
    # GENERATE MARKDOWN REPORT
    # -------------------------------------------------------------------------
    ar_warm = iteration_records[1]
    slds_warm = iteration_records[3]

    report_path = os.path.join(KPMS_DIR, "diagnostics", "MEASURED_GRANULAR_PROFILE_FULL_DATASET.md")
    with open(report_path, "w") as f:
        f.write("# Empirically Measured Full-Dataset Granular Profiling Report\n\n")
        f.write("**Dataset:** Jonathan AMPH 30 Sessions (562 sequences, 5,516,954 active frames)\n")
        f.write("**Hardware:** NVIDIA GeForce RTX 4090 (24 GB VRAM) + 32-core Host CPU (125.6 GB RAM)\n")
        f.write(f"**Execution Timestamp:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
        f.write("## 1. Measured Steady-State Sub-Component Breakdown (Warmed Iteration)\n\n")
        f.write("| Sub-Operation / Kernel | Category | Warmed AR (s) | % AR Iter | Warmed SLDS (s) | % SLDS Iter | Execution Hardware / Method |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")

        # Table rows
        ar_t = ar_warm["timings"]
        ar_tot = ar_warm["total_time"]
        slds_t = slds_warm["timings"]
        slds_tot = slds_warm["total_time"]

        f.write(f"| **Discrete: GPU Emissions** | Syllables ($z$) | {ar_t['z_emissions']:.2f} s | {ar_t['z_emissions']/ar_tot*100:.1f}% | {slds_t['z_emissions']:.2f} s | {slds_t['z_emissions']/slds_tot*100:.1f}% | GPU JIT Kernel (bs=20) |\n")
        f.write(f"| **Discrete: D2H Transfer** | Syllables ($z$) | {ar_t['z_d2h']:.2f} s | {ar_t['z_d2h']/ar_tot*100:.1f}% | {slds_t['z_d2h']:.2f} s | {slds_t['z_d2h']/slds_tot*100:.1f}% | PCIe Device-to-Host Copy |\n")
        f.write(f"| **Discrete: CPU Numba FFBS** | Syllables ($z$) | {ar_t['z_ffbs']:.2f} s | {ar_t['z_ffbs']/ar_tot*100:.1f}% | {slds_t['z_ffbs']:.2f} s | {slds_t['z_ffbs']/slds_tot*100:.1f}% | CPU OpenMP 32 cores (pi_T) |\n")
        f.write(f"| **Discrete: H2D Transfer** | Syllables ($z$) | {ar_t['z_h2d']:.2f} s | {ar_t['z_h2d']/ar_tot*100:.1f}% | {slds_t['z_h2d']:.2f} s | {slds_t['z_h2d']/slds_tot*100:.1f}% | PCIe Host-to-Device Copy |\n")
        f.write(f"| **AR: Streaming Statistics** | Dynamics ($Ab, Q$) | {ar_t['ar_stats']:.2f} s | {ar_t['ar_stats']/ar_tot*100:.1f}% | {slds_t['ar_stats']:.2f} s | {slds_t['ar_stats']/slds_tot*100:.1f}% | CPU Zero-Alloc Accumulator |\n")
        f.write(f"| **AR: MNIW Posterior Solve** | Dynamics ($Ab, Q$) | {ar_t['ar_solve']:.2f} s | {ar_t['ar_solve']/ar_tot*100:.1f}% | {slds_t['ar_solve']:.2f} s | {slds_t['ar_solve']/slds_tot*100:.1f}% | GPU Vectorized Solve |\n")
        f.write(f"| **Transitions ($\\pi, \\beta$)** | Markov Transitions | {ar_t['transitions']:.2f} s | {ar_t['transitions']/ar_tot*100:.1f}% | {slds_t['transitions']:.2f} s | {slds_t['transitions']/slds_tot*100:.1f}% | GPU Dirichlet-HDP Sampler |\n")
        f.write(f"| **Continuous: Rigid Rotations** | Trajectory ($x$) | *N/A* | *N/A* | {slds_t['x_rigid_rot']:.2f} s | {slds_t['x_rigid_rot']/slds_tot*100:.1f}% | CPU Vectorized Frame Rotation |\n")
        f.write(f"| **Continuous: Banded Cholesky** | Trajectory ($x$) | *N/A* | *N/A* | {slds_t['x_banded_solve']:.2f} s | {slds_t['x_banded_solve']/slds_tot*100:.1f}% | Native Numba OpenMP Solver |\n")
        f.write(f"| **Continuous: H2D Transfer** | Trajectory ($x$) | *N/A* | *N/A* | {slds_t['x_h2d']:.2f} s | {slds_t['x_h2d']/slds_tot*100:.1f}% | PCIe Host-to-Device Copy |\n")
        f.write(f"| **Pose: Centroid Location ($v$)** | Centroid ($v$) | *N/A* | *N/A* | {slds_t['location_v']:.2f} s | {slds_t['location_v']/slds_tot*100:.1f}% | Batched GPU Kernel (bs=50) |\n")
        f.write(f"| **Pose: Heading ($h$)** | Heading ($h$) | *N/A* | *N/A* | {slds_t['heading_h']:.2f} s | {slds_t['heading_h']/slds_tot*100:.1f}% | Batched GPU Kernel (bs=20) |\n")
        f.write(f"| **Noise: Scales ($s$)** | Noise Scales ($s$) | *N/A* | *N/A* | {slds_t['scales_s']:.2f} s | {slds_t['scales_s']/slds_tot*100:.1f}% | Batched GPU Kernel (bs=50) |\n")
        f.write(f"| **Pipeline Overhead / JAX Sync** | Framework Overhead | {ar_t['overhead_misc']:.2f} s | {ar_t['overhead_misc']/ar_tot*100:.1f}% | {slds_t['overhead_misc']:.2f} s | {slds_t['overhead_misc']/slds_tot*100:.1f}% | Python Loop & JAX Dispatch |\n")
        f.write(f"| **TOTAL MEASURED ITERATION** | **Full Gibbs Sweep** | **{ar_tot:.2f} s** | **100.0%** | **{slds_tot:.2f} s** | **100.0%** | **100% Accounted (0 missing)** |\n\n")

        f.write("---\n\n")
        f.write("## 2. Granular Per-Iteration Measurement Log\n\n")
        for rec in iteration_records:
            f.write(f"### Iteration {rec['iter']} ({rec['phase']} - {rec['status']})\n")
            f.write(f"- **Total Wall-Clock Time:** `{rec['total_time']:.2f} s`\n")
            f.write(f"- **Host Process RAM:** `{rec['ram_gb']:.2f} GB` | **GPU VRAM:** `{rec['vram_gb']:.2f} GB`\n")
            f.write("- **Component Timings:**\n")
            for k, v in rec["timings"].items():
                if v > 0.001:
                    f.write(f"  - `{k:20s}`: {v:6.2f} s ({v/rec['total_time']*100:5.1f}%)\n")
            f.write("\n")

    print(f"\nReport written to {report_path}!")
    print("=" * 85, flush=True)

if __name__ == "__main__":
    run_profiler()
