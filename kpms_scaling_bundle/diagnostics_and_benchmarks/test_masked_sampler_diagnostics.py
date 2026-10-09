import os
import sys

import numpy as np

# Numpy 2.x compatibility
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

REPO_ROOT = "/home/fsp585/Desktop/FiberphotometryCode"
KPMS_DIR = os.path.join(REPO_ROOT, "syllable_generation", "syllabizers", "keypoint_moseq_syll")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if KPMS_DIR not in sys.path:
    sys.path.insert(0, KPMS_DIR)

import jax.numpy as jnp
import jax.random as jr
import jax_moseq.models.keypoint_slds.alignment as slds_alignment
import jax_moseq.models.keypoint_slds.gibbs as slds_gibbs
from numba import njit, prange
from utils.resample_batching import (
    CONFIG,
    _numba_banded_backward_solve,
    _numba_banded_cholesky_inplace,
    _numba_banded_forward_solve_inplace,
    _numba_fused_rigid_transform,
    pure_numba_resample_continuous_stateseqs,
)


# -----------------------------------------------------------------------------
# Custom Numba kernel that includes upstream pseudo-observations on masked frames
# -----------------------------------------------------------------------------
@njit(fastmath=True, parallel=True)
def _numba_assemble_and_solve_with_pseudo_obs(
    ab_all, g_all, eta_all, x_all,
    Y_np, mask_np, s_np, d_off,
    C_mat, sigmasq_np, K_blocks, g_priors, z_np,
    N_seq, T, D, nlags, bandwidth, block_dim, N_total, jitter,
    repairs_all,
    pseudo_obs_cov=10.0
):
    masked_inv_cov = 0.1  # 1 / 10.0 dynamics noise for masked frames
    inv_pseudo_obs = 1.0 / pseudo_obs_cov

    for seq_i in prange(N_seq):
        ab = ab_all[seq_i]
        g = g_all[seq_i]
        eta = eta_all[seq_i]
        x_flat = x_all[seq_i]

        # 1. Initial state prior on first nlags frames (S0 = 10 * I -> precision = 0.1)
        for d in range(nlags * D):
            ab[0, d] += 0.1

        # 2. Observations
        for t in range(T):
            t_idx = t * D
            is_valid = mask_np[seq_i, t]
            for o_idx in range(C_mat.shape[0]):
                if is_valid:
                    inv_R = 1.0 / (sigmasq_np[o_idx] * s_np[seq_i, t, o_idx] + 1e-8)
                else:
                    inv_R = inv_pseudo_obs  # Upstream uses masked_obs_noise = 10.0

                y_diff = Y_np[seq_i, t, o_idx] - d_off[o_idx]
                for r in range(D):
                    c_ir = C_mat[o_idx, r]
                    if c_ir != 0.0:
                        g[t_idx + r] += c_ir * inv_R * y_diff
                        for c in range(r, D):
                            c_ic = C_mat[o_idx, c]
                            if c_ic != 0.0:
                                ab[c - r, t_idx + r] += c_ir * inv_R * c_ic

        # 3. Dynamics transitions
        for t in range(nlags, T):
            is_valid = mask_np[seq_i, t]
            start_idx = (t - nlags) * D

            if is_valid:
                z_t = z_np[seq_i, t - nlags]
                K_t = K_blocks[z_t]
                g_t = g_priors[z_t]

                for r in range(block_dim):
                    for c in range(r, block_dim):
                        band_idx = c - r
                        if band_idx <= bandwidth:
                            ab[band_idx, start_idx + r] += K_t[r, c]
                    g[start_idx + r] += g_t[r]
            else:
                t_idx = t * D
                prev_idx = (t - 1) * D
                for d in range(D):
                    ab[0, t_idx + d] += masked_inv_cov
                    ab[0, prev_idx + d] += masked_inv_cov
                    if D <= bandwidth:
                        ab[D, prev_idx + d] -= masked_inv_cov

        # 4. Regularization jitter on diagonal
        for idx in range(N_total):
            ab[0, idx] += jitter

        # 5. In-place Banded Cholesky
        repairs_all[seq_i] = _numba_banded_cholesky_inplace(ab, bandwidth, N_total)

        # 6. In-place forward solve
        _numba_banded_forward_solve_inplace(ab, bandwidth, N_total, g)

        # 7. Add Gaussian noise eta directly and backward solve
        for idx in range(N_total):
            g[idx] += eta[idx]
        _numba_banded_backward_solve(ab, bandwidth, N_total, g, x_flat)


def numba_resample_continuous_with_pseudo_obs(
    seed, Y, mask, v, h, s, z, Cd, sigmasq, Ab, Q, jitter=1e-3, pseudo_obs_cov=10.0, **kwargs
):
    N, T, k, d = Y.shape
    num_states, D, ar_dim_plus_1 = Ab.shape
    nlags = (ar_dim_plus_1 - 1) // D
    bandwidth = (nlags + 1) * D - 1
    block_dim = (nlags + 1) * D
    N_total = T * D

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

    CHUNK_SIZE = min(kwargs.get("chunk_size", CONFIG.continuous_chunk_size), N)
    ab_workspace = np.zeros((CHUNK_SIZE, bandwidth + 1, N_total), dtype=np.float64)
    g_workspace = np.zeros((CHUNK_SIZE, N_total), dtype=np.float64)
    eta_workspace = np.zeros((CHUNK_SIZE, N_total), dtype=np.float64)
    x_workspace = np.zeros((CHUNK_SIZE, N_total), dtype=np.float64)
    repairs_workspace = np.zeros(CHUNK_SIZE, dtype=np.int64)
    Y_rot_workspace = np.empty((CHUNK_SIZE, T, k * d), dtype=np.float64)

    x_out = np.zeros((N, T, D), dtype=np.float64)
    mask_np = np.asarray(mask, dtype=bool)
    z_np = np.asarray(z, dtype=np.int32)
    Y_master_cpu = np.asarray(Y, dtype=np.float64)

    for chunk_start in range(0, N, CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, N)
        N_chunk = chunk_end - chunk_start

        Y_c = Y_master_cpu[chunk_start:chunk_end]
        v_c = np.asarray(v[chunk_start:chunk_end], dtype=np.float64)
        h_c = np.asarray(h[chunk_start:chunk_end], dtype=np.float64)
        s_c = np.asarray(s[chunk_start:chunk_end], dtype=np.float64)

        Y_c_van = Y_rot_workspace[:N_chunk]
        _numba_fused_rigid_transform(Y_c, v_c, h_c, Y_c_van)
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

        _numba_assemble_and_solve_with_pseudo_obs(
            ab_chunk, g_chunk, eta_chunk, x_chunk,
            Y_c_van,
            mask_np[chunk_start:chunk_end],
            s_c_van,
            d_off, C_mat, sigmasq_np,
            K_blocks, g_priors,
            z_np[chunk_start:chunk_end],
            N_chunk, T, D, nlags, bandwidth, block_dim, N_total, jitter,
            repairs_chunk,
            pseudo_obs_cov=pseudo_obs_cov
        )
        x_out[chunk_start:chunk_end] = x_chunk.reshape(N_chunk, T, D)

    return jnp.asarray(x_out)


def run_diagnostics():
    print("=" * 80, flush=True)
    print("  CONTROLLED 4-CONDITION CONTINUOUS SAMPLER DIAGNOSTIC", flush=True)
    print("=" * 80, flush=True)

    # 1. Load checkpoint
    ckpt_path = os.path.join(KPMS_DIR, "amph_project", "amph_prod_k150k_run_01")
    if not os.path.exists(os.path.join(ckpt_path, "checkpoint.h5")):
        ckpt_path = os.path.join(KPMS_DIR, "amph_project", "amph_benchmark_test_timing")
    print(f"Loading checkpoint from {ckpt_path}...", flush=True)

    import h5py
    with h5py.File(os.path.join(ckpt_path, "checkpoint.h5"), "r") as f:
        # Load snapshot 100 or latest
        if "model_snapshots" in f:
            snaps = sorted([int(k) for k in f["model_snapshots"].keys()])
            target_snap = str(snaps[-1])
            model_grp = f["model_snapshots"][target_snap]
        else:
            model_grp = f["model"]

        Cd = jnp.array(model_grp["params"]["Cd"][:])
        sigmasq = jnp.array(model_grp["params"]["sigmasq"][:])
        Ab = jnp.array(model_grp["params"]["Ab"][:])
        Q = jnp.array(model_grp["params"]["Q"][:])

        v_slice = jnp.array(model_grp["states"]["v"][:2, :400])
        h_slice = jnp.array(model_grp["states"]["h"][:2, :400])
        s_slice = jnp.array(model_grp["states"]["s"][:2, :400])
        z_slice = jnp.array(model_grp["states"]["z"][:2, :400 - 3])

        Y_slice = jnp.array(f["data"]["Y"][:2, :400])
        mask_raw = np.array(f["data"]["mask"][:2, :400]).astype(bool)

    N, T, k, d = Y_slice.shape
    D = Ab.shape[1]
    print(f"Test slice: N={N}, T={T}, k={k}, D={D}", flush=True)

    # Inject 15-frame gap at [100:115]
    mask_test = np.ones((N, T), dtype=bool)
    mask_test[:, 100:115] = False
    mask_test_jnp = jnp.array(mask_test)

    N_DRAWS = 50
    base_seed = 42

    print(f"\nRunning {N_DRAWS} Monte Carlo draws per condition across 15-frame gap [100:115]...\n", flush=True)

    # Condition 1: Baseline Upstream JAX Kalman (masked_obs_noise = 10.0)
    print("[Condition 1] Upstream JAX Kalman (masked_obs_noise = 10.0)...", flush=True)
    c1_samples = np.zeros((N_DRAWS, N, T, D))
    for d_i in range(N_DRAWS):
        key = jr.PRNGKey(base_seed + d_i)
        x = slds_gibbs.resample_continuous_stateseqs(
            key, Y_slice, mask_test_jnp, v_slice, h_slice, s_slice,
            z_slice, Cd, sigmasq, Ab, Q, jitter=1e-3, masked_obs_noise=10.0,
            parallel_message_passing=False
        )
        x.block_until_ready()
        c1_samples[d_i] = np.array(x)

    # Condition 2: Baseline Numba Banded Cholesky (zero emissions in gap, pure random walk)
    print("[Condition 2] Baseline Numba Banded Cholesky (zero emissions, pure random walk)...", flush=True)
    c2_samples = np.zeros((N_DRAWS, N, T, D))
    for d_i in range(N_DRAWS):
        key = jr.PRNGKey(base_seed + 1000 + d_i)
        x = pure_numba_resample_continuous_stateseqs(
            key, Y_slice, mask_test_jnp, v_slice, h_slice, s_slice,
            z_slice, Cd, sigmasq, Ab, Q, jitter=1e-3
        )
        c2_samples[d_i] = np.array(x)

    # Condition 3: Numba Matched to Upstream (pseudo-observation cov = 10.0 on masked frames)
    print("[Condition 3] Numba Matched to Upstream (pseudo-obs cov = 10.0 in gap)...", flush=True)
    c3_samples = np.zeros((N_DRAWS, N, T, D))
    for d_i in range(N_DRAWS):
        key = jr.PRNGKey(base_seed + 2000 + d_i)
        x = numba_resample_continuous_with_pseudo_obs(
            key, Y_slice, mask_test_jnp, v_slice, h_slice, s_slice,
            z_slice, Cd, sigmasq, Ab, Q, jitter=1e-3, pseudo_obs_cov=10.0
        )
        c3_samples[d_i] = np.array(x)

    # Condition 4: Upstream JAX Kalman Matched to True Missingness (masked_obs_noise = 1e6)
    print("[Condition 4] Upstream JAX Kalman with True Missingness (masked_obs_noise = 1e6)...", flush=True)
    c4_samples = np.zeros((N_DRAWS, N, T, D))
    for d_i in range(N_DRAWS):
        key = jr.PRNGKey(base_seed + 3000 + d_i)
        x = slds_gibbs.resample_continuous_stateseqs(
            key, Y_slice, mask_test_jnp, v_slice, h_slice, s_slice,
            z_slice, Cd, sigmasq, Ab, Q, jitter=1e-3, masked_obs_noise=1e6,
            parallel_message_passing=False
        )
        x.block_until_ready()
        c4_samples[d_i] = np.array(x)

    # Compute Variances
    var_c1 = np.var(c1_samples, axis=0)
    var_c2 = np.var(c2_samples, axis=0)
    var_c3 = np.var(c3_samples, axis=0)
    var_c4 = np.var(c4_samples, axis=0)

    # Gap window: [100:115], Valid window: [50:90]
    gap_slice = slice(100, 115)
    val_slice = slice(50, 90)

    v1_gap, v1_val = float(np.mean(var_c1[:, gap_slice])), float(np.mean(var_c1[:, val_slice]))
    v2_gap, v2_val = float(np.mean(var_c2[:, gap_slice])), float(np.mean(var_c2[:, val_slice]))
    v3_gap, v3_val = float(np.mean(var_c3[:, gap_slice])), float(np.mean(var_c3[:, val_slice]))
    v4_gap, v4_val = float(np.mean(var_c4[:, gap_slice])), float(np.mean(var_c4[:, val_slice]))

    # Midpoint of gap (frame 107)
    mid_c1 = float(np.mean(var_c1[:, 107]))
    mid_c2 = float(np.mean(var_c2[:, 107]))
    mid_c3 = float(np.mean(var_c3[:, 107]))
    mid_c4 = float(np.mean(var_c4[:, 107]))

    # Trajectory correlations against Condition 1
    m1 = np.mean(c1_samples, axis=0)
    m2 = np.mean(c2_samples, axis=0)
    m3 = np.mean(c3_samples, axis=0)
    m4 = np.mean(c4_samples, axis=0)

    r_c2_c1 = np.mean([np.corrcoef(m1[:, :, d].ravel(), m2[:, :, d].ravel())[0, 1] for d in range(D)])
    r_c3_c1 = np.mean([np.corrcoef(m1[:, :, d].ravel(), m3[:, :, d].ravel())[0, 1] for d in range(D)])

    print("\n" + "=" * 90, flush=True)
    print(f"{'4-CONDITION CONTINUOUS LATENT SAMPLER COMPARISON RESULTS':^90}", flush=True)
    print("=" * 90, flush=True)
    header = f"{'Condition':<42} | {'Valid Var':<10} | {'Gap Var':<10} | {'Expansion':<10} | {'Midpoint Var':<12}"
    print(header, flush=True)
    print("-" * 90, flush=True)
    print(f"{'1. Upstream Kalman (cov=10)':<42} | {v1_val:<10.4f} | {v1_gap:<10.4f} | {v1_gap/v1_val:<10.2f}x | {mid_c1:<12.4f}", flush=True)
    print(f"{'2. Baseline Numba (cov=inf, pure RW)':<42} | {v2_val:<10.4f} | {v2_gap:<10.4f} | {v2_gap/v2_val:<10.2f}x | {mid_c2:<12.4f}", flush=True)
    print(f"{'3. Numba Matched (cov=10 pseudo-obs)':<42} | {v3_val:<10.4f} | {v3_gap:<10.4f} | {v3_gap/v3_val:<10.2f}x | {mid_c3:<12.4f}", flush=True)
    print(f"{'4. Kalman Matched (cov=1e6 true missing)':<42} | {v4_val:<10.4f} | {v4_gap:<10.4f} | {v4_gap/v4_val:<10.2f}x | {mid_c4:<12.4f}", flush=True)
    print("=" * 90, flush=True)

    print("\nTrajectory Correlation to Upstream Kalman (Baseline):", flush=True)
    print(f"  * Baseline Numba (Condition 2 vs 1):  r = {r_c2_c1:.4f}", flush=True)
    print(f"  * Matched Numba  (Condition 3 vs 1):  r = {r_c3_c1:.4f}", flush=True)
    print("=" * 90, flush=True)

if __name__ == "__main__":
    run_diagnostics()
