import os
import sys

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

np.bool8 = np.bool_
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

REPO_ROOT = "/home/fsp585/Desktop/FiberphotometryCode"
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
KPMS_DIR = os.path.join(REPO_ROOT, "syllable_generation", "syllabizers", "keypoint_moseq_syll")
if KPMS_DIR not in sys.path:
    sys.path.insert(0, KPMS_DIR)

import jax_moseq.models.arhmm.gibbs as arhmm_gibbs
import jax_moseq.models.keypoint_slds.gibbs as slds_gibbs
import keypoint_moseq as kpms
from jax_moseq.utils import pad_affine, psd_inv, psd_solve
from jax_moseq.utils.autoregression import get_lags
from utils.batch_init import batched_init_model_by_video
from utils.paths import ref_pairs_from_sessions
from utils.trimming import clamp_conf_nonneg, filter_with_masks, presence_mask_from_conf

SCRATCH_DIR = os.path.join(KPMS_DIR, "scratch_profiling_project")
DIRTY_PKL = os.path.join(REPO_ROOT, "syllable_generation/data/sessions_karen_project_dirty.pkl")

# Custom GPU Vectorized AR
def custom_ar_sample(seed, mask, x, z, **ar_hypparams):
    nlags = ar_hypparams.get("nlags", mask.shape[1] - z.shape[1])
    num_states = ar_hypparams.get("num_states", 100)
    dtype = x.dtype
    seeds = jr.split(seed, num_states)
    mask_flat = mask[..., nlags:].reshape(-1)
    z_flat = z.reshape(-1)
    x_in = pad_affine(get_lags(x, nlags)).reshape(-1, nlags * x.shape[-1] + 1)
    x_out = x[..., nlags:, :].reshape(-1, x.shape[-1])

    K_0_inv = psd_inv(ar_hypparams["K_0"].astype(dtype))
    nu_0 = ar_hypparams["nu_0"]
    S_0 = ar_hypparams["S_0"].astype(dtype)
    M_0 = ar_hypparams["M_0"].astype(dtype)

    def sample_one_state(seed_k, k_idx):
        m_k = mask_flat * (z_flat == k_idx)
        x_out_m = x_out * m_k[:, None]
        S_out_out = x_out_m.T @ x_out
        S_out_in = x_out_m.T @ x_in
        x_in_m = x_in * m_k[:, None]
        S_in_in = x_in_m.T @ x_in

        K_n_inv = K_0_inv + S_in_in
        K_n = psd_inv(K_n_inv)
        M_n = psd_solve(K_n_inv.T, K_0_inv @ M_0.T + S_out_in.T).T
        S_n = S_0 + S_out_out + (M_0 @ K_0_inv @ M_0.T - M_n @ K_n_inv @ M_n.T)
        return arhmm_gibbs.sample_mniw(seed_k, nu_0 + m_k.sum(), S_n, M_n, K_n)

    return jax.vmap(sample_one_state)(seeds, jnp.arange(num_states))


# Custom Single-Pass Tensor-Core Discrete Stateseq Resampler with Gumbel argmax
def custom_z_sample(seed, x, mask, Ab, Q, pi, nlags=3, num_states=100):
    N, T, D = x.shape
    dtype = x.dtype
    eps = jnp.array(1e-12, dtype=dtype)
    L_Q = jnp.linalg.cholesky(Q)
    half_log_det_Q = jnp.sum(jnp.log(jnp.diagonal(L_Q, axis1=-2, axis2=-1)), axis=-1)
    const = (jnp.array(0.5 * D * np.log(2.0 * np.pi), dtype=dtype)) + half_log_det_Q
    Q_inv = jax.vmap(psd_inv)(Q)

    lags = get_lags(x, nlags)
    ones = jnp.ones((*lags.shape[:-1], 1), dtype=dtype)
    x_lags = jnp.concatenate([lags, ones], axis=-1)
    x_target = x[:, nlags:, :]

    mu_all = jnp.einsum("nti,kdi->ntkd", x_lags, Ab)
    diff = x_target[:, :, None, :] - mu_all
    quad = jnp.einsum("ntkd,kde,ntke->ntk", diff, Q_inv, diff)
    log_likelihoods = (-0.5 * quad - const[None, None, :]).astype(dtype)

    log_A = jnp.log(pi.astype(dtype) + eps).astype(dtype)
    log_init = jnp.log(jnp.ones(num_states, dtype=dtype) / jnp.array(num_states, dtype=dtype)).astype(dtype)

    def sample_one(seed_i, ll_i, mask_i):
        def fwd_step(carry, t):
            log_pred = carry
            ll_t = (ll_i[t] * mask_i[nlags + t]).astype(dtype)
            log_filt = log_pred + ll_t
            log_norm = jnp.max(log_filt) + jnp.log(jnp.sum(jnp.exp(log_filt - jnp.max(log_filt))))
            log_filt = log_filt - log_norm
            matrix_term = log_filt[:, None] + log_A
            max_m = jnp.max(matrix_term, axis=0)
            log_pred_next = max_m + jnp.log(jnp.sum(jnp.exp(matrix_term - max_m[None, :]), axis=0))
            return log_pred_next.astype(dtype), log_filt.astype(dtype)

        T_z = ll_i.shape[0]
        _, log_filts = jax.lax.scan(fwd_step, log_init, jnp.arange(T_z))

        gumbel_key, _ = jr.split(seed_i)
        gumbels = jr.gumbel(gumbel_key, shape=(T_z, num_states), dtype=dtype)
        last_z = jnp.argmax(log_filts[-1] + gumbels[-1])

        def bwd_step(next_z, t):
            scores = log_filts[t] + log_A[:, next_z] + gumbels[t]
            curr_z = jnp.argmax(scores)
            return curr_z, curr_z

        _, z_seq = jax.lax.scan(bwd_step, last_z, jnp.arange(T_z - 2, -1, -1))
        return jnp.concatenate([z_seq[::-1], last_z[None]])

    return jax.vmap(sample_one)(jr.split(seed, N), log_likelihoods, mask)


def test_multi_iter_drift(n_iters: int = 25):
    print("=" * 80)
    print(f"      MULTI-ITERATION GIBBS STABILITY TEST ({n_iters} ITERATIONS)")
    print("=" * 80)

    jax.config.update("jax_enable_x64", True)

    sessions = pd.read_pickle(DIRTY_PKL)[:1]
    keypoint_paths, video_paths = ref_pairs_from_sessions(sessions, subroutine="jonathan", cam="center_cam")
    coordinates, confidences, _ = kpms.load_keypoints(keypoint_paths, "deeplabcut")
    all_masks = {name: presence_mask_from_conf(conf) for name, conf in confidences.items()}
    filtered_coords, filtered_confs, _ = filter_with_masks(
        coordinates, confidences, all_masks, SCRATCH_DIR,
        return_video_frame_indexes=True, downsample_rate=2,
        **kpms.load_config(SCRATCH_DIR)
    )
    clamp_conf_nonneg(filtered_confs)
    data_raw, _ = kpms.format_data(filtered_coords, filtered_confs, **kpms.load_config(SCRATCH_DIR))
    data_raw = {k: v[:2] for k, v in data_raw.items()}
    pca = kpms.fit_pca(**data_raw, **kpms.load_config(SCRATCH_DIR))

    init_config = kpms.load_config(SCRATCH_DIR)
    base_model = batched_init_model_by_video(data_raw, pca, init_config, params_subset_size=2, batch_size=20, seed=jr.PRNGKey(42))
    base_model = kpms.update_hypparams(base_model, kappa=6000.0)

    def run_chain(precision_mode: str):
        print(f"\n--- Running Chain [{precision_mode}] for {n_iters} Gibbs iterations ---")
        dtype = jnp.float32 if precision_mode == "FP32" else jnp.float64

        def cast(tree):
            return jax.tree_util.tree_map(lambda x: x.astype(dtype) if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating) else x, tree)

        data = cast(data_raw)
        model = cast(base_model)
        seed = jr.PRNGKey(42)
        noise_prior = model.get("noise_prior", 1.0)

        history = []
        for it in range(1, n_iters + 1):
            seed, s_iter = jr.split(seed)
            # 1. HDP
            model["params"]["betas"], model["params"]["pi"] = arhmm_gibbs.resample_hdp_transitions(
                s_iter, **data, **model["states"], **model["params"], **model["hypparams"]["trans_hypparams"]
            )
            # 2. AR (custom GPU vectorized)
            model["params"]["Ab"], model["params"]["Q"] = custom_ar_sample(
                s_iter, data["mask"], model["states"]["x"], model["states"]["z"], **model["hypparams"]["ar_hypparams"]
            )
            # 3. z (custom Gumbel argmax)
            model["states"]["z"] = custom_z_sample(
                s_iter, model["states"]["x"], data["mask"], model["params"]["Ab"], model["params"]["Q"], model["params"]["pi"]
            )
            # 4. x
            model["states"]["x"] = slds_gibbs.resample_continuous_stateseqs(
                s_iter, **data, **model["states"], **model["params"], jitter=1e-3, parallel_message_passing=False
            )
            # 5. h & v
            model["states"]["h"] = slds_gibbs.resample_heading(s_iter, **data, **model["states"], **model["params"])
            model["states"]["v"] = slds_gibbs.resample_location(s_iter, **data, **model["states"], **model["params"], **model["hypparams"]["cen_hypparams"])
            # 6. s
            model["states"]["s"] = slds_gibbs.resample_scales(s_iter, **data, **model["states"], **model["params"], s_0=noise_prior, **model["hypparams"]["obs_hypparams"])

            model["states"]["x"].block_until_ready()

            x_arr = np.array(model["states"]["x"])
            has_nan = np.isnan(x_arr).any() or np.isnan(np.array(model["params"]["Ab"])).any()
            max_x = float(np.max(np.abs(x_arr))) if not has_nan else float('nan')
            mean_ab = float(np.mean(np.linalg.norm(np.array(model["params"]["Ab"]), axis=(1, 2)))) if not has_nan else float('nan')
            sylls = len(np.unique(np.array(model["states"]["z"]))) if not has_nan else 0

            history.append({"iter": it, "max_x": max_x, "mean_ab": mean_ab, "sylls": sylls, "has_nan": has_nan})
            status = "DIVERGED/NaN" if has_nan else f"max|x|={max_x:5.2f}, ||Ab||={mean_ab:5.2f}, sylls={sylls:2d}"
            if it % 5 == 0 or it == n_iters or has_nan:
                print(f"  Iter {it:2d}/{n_iters}: {status}")
            if has_nan:
                print(f"  [CRITICAL] Divergence triggered at iteration {it} in {precision_mode} mode.")
                break

        return history

    h_64 = run_chain("FP64")
    h_32 = run_chain("FP32")

    print("\n" + "=" * 80)
    print("                  FINAL MULTI-ITERATION STABILITY COMPARISON")
    print("=" * 80)
    print(f"{'Iteration':<10} | {'FP64 Max |x|':<14} | {'FP32 Max |x|':<14} | {'FP64 Sylls':<12} | {'FP32 Sylls':<12}")
    print("-" * 80)
    for i in range(min(len(h_64), len(h_32))):
        it = h_64[i]["iter"]
        if it % 5 == 0 or it == n_iters or i == len(h_32)-1:
            print(f"{it:<10} | {h_64[i]['max_x']:12.4f}   | {h_32[i]['max_x']:12.4f}   | {h_64[i]['sylls']:10d}   | {h_32[i]['sylls']:10d}")
    print("=" * 80)

if __name__ == "__main__":
    test_multi_iter_drift(25)
