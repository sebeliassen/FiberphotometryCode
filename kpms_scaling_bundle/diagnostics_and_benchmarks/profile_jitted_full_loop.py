import os
import sys
import time

import numpy as np

np.bool8 = np.bool_
from functools import partial

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

import jax_moseq.models.keypoint_slds.gibbs as slds_gibbs
import keypoint_moseq as kpms
from jax_moseq.utils.autoregression import get_nlags
from jax_moseq.utils.transitions import count_transitions, sample_hdp_transitions
from utils.batch_init import batched_init_model_by_video
from utils.paths import ref_pairs_from_sessions
from utils.resample_batching import (
    _numba_assemble_and_solve_tridiagonal_v,
    fast_banded_gmrf_resample_x,
    fast_resample_discrete_stateseqs,
    vectorized_resample_ar_params,
)
from utils.trimming import clamp_conf_nonneg, filter_with_masks, presence_mask_from_conf

SCRATCH_DIR = os.path.join(KPMS_DIR, "scratch_profiling_project")
DIRTY_PKL = os.path.join(REPO_ROOT, "syllable_generation/data/sessions_karen_project_dirty.pkl")

# JIT-compiled pose & scales functions
@jax.jit
def jit_resample_heading(seed, Y, x, v, s, Cd, sigmasq):
    return slds_gibbs.resample_heading(seed, Y, x, v, s, Cd, sigmasq)

@partial(jax.jit, static_argnums=(2,))
def jit_pose_prep(x, Cd, k, h):
    Y_aligned = slds_gibbs.estimate_aligned(x, Cd, k)
    return slds_gibbs.apply_rotation(Y_aligned, h)

@jax.jit
def jit_resample_scales(seed, Y, x, v, h, Cd, sigmasq, mask, s_0, nu_s):
    return slds_gibbs.resample_scales(seed, Y=Y, x=x, v=v, h=h, Cd=Cd, sigmasq=sigmasq, mask=mask, s_0=s_0, nu_s=nu_s)

def main():
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
    data, _ = kpms.format_data(filtered_coords, filtered_confs, **kpms.load_config(SCRATCH_DIR))
    data = {k: v[:2] for k, v in data.items()}
    pca = kpms.fit_pca(**data, **kpms.load_config(SCRATCH_DIR))

    seed = jr.PRNGKey(42)
    init_config = kpms.load_config(SCRATCH_DIR)
    model = batched_init_model_by_video(data, pca, init_config, params_subset_size=2, batch_size=20, seed=seed)
    model = kpms.update_hypparams(model, kappa=6000.0)

    states = model["states"]
    params = model["params"]
    hypparams = model["hypparams"]
    Y = data["Y"]
    mask = data["mask"]

    N, T, k, d = Y.shape
    nlags = get_nlags(params["Ab"])
    num_states = len(params["betas"])
    sigmasq_loc = float(hypparams["cen_hypparams"]["sigmasq_loc"])

    print("=" * 80)
    print("      FULLY JIT-COMPILED GIBBS ITERATION PROFILER")
    print("=" * 80)

    # Warmup all JIT kernels
    print("Warming up all JIT kernels...")
    _ = fast_resample_discrete_stateseqs(seed, states["x"], mask, params["Ab"], params["Q"], params["pi"], nlags=nlags, num_states=num_states)
    _ = fast_banded_gmrf_resample_x(seed, Y=Y, mask=mask, v=states["v"], h=states["h"], s=states["s"], z=states["z"], Cd=params["Cd"], sigmasq=params["sigmasq"], Ab=params["Ab"], Q=params["Q"], nlags=nlags)
    _ = jit_resample_heading(seed, Y, states["x"], states["v"], states["s"], params["Cd"], params["sigmasq"])
    _ = jit_pose_prep(states["x"], params["Cd"], k, states["h"])
    _ = jit_resample_scales(seed, Y, states["x"], states["v"], states["h"], params["Cd"], params["sigmasq"], mask, model.get("noise_prior", 1.0), float(hypparams["obs_hypparams"]["nu_s"]))
    _ = vectorized_resample_ar_params(seed, x=states["x"], z=states["z"], mask=mask, **hypparams["ar_hypparams"])
    print("Warmup complete.\n")

    for it in range(1, 6):
        t_tot = time.perf_counter()

        # 1. Transitions
        t0 = time.perf_counter()
        counts = count_transitions(num_states, states["z"], mask)
        params["betas"], params["pi"] = sample_hdp_transitions(
            jr.fold_in(seed, it * 20 + 1), counts,
            params["betas"], hypparams["trans_hypparams"]["alpha"],
            hypparams["trans_hypparams"]["kappa"], hypparams["trans_hypparams"]["gamma"]
        )
        params["pi"].block_until_ready()
        t_trans = (time.perf_counter() - t0) * 1000.0

        # 2. AR Dynamics (Ab, Q)
        t0 = time.perf_counter()
        params["Ab"], params["Q"] = vectorized_resample_ar_params(
            jr.fold_in(seed, it * 20 + 2), x=states["x"], z=states["z"], mask=mask, **hypparams["ar_hypparams"]
        )
        params["Q"].block_until_ready()
        t_ar = (time.perf_counter() - t0) * 1000.0

        # 3. Discrete Syllables (z)
        t0 = time.perf_counter()
        states["z"] = fast_resample_discrete_stateseqs(
            jr.fold_in(seed, it * 20 + 3), states["x"], mask, params["Ab"], params["Q"], params["pi"],
            nlags=nlags, num_states=num_states
        )
        states["z"].block_until_ready()
        t_z = (time.perf_counter() - t0) * 1000.0

        # 4. Heading (h)
        t0 = time.perf_counter()
        states["h"] = jit_resample_heading(
            jr.fold_in(seed, it * 20 + 4), Y, states["x"], states["v"], states["s"], params["Cd"], params["sigmasq"]
        )
        states["h"].block_until_ready()
        t_h = (time.perf_counter() - t0) * 1000.0

        # 5. Centroid Location (v)
        t0 = time.perf_counter()
        Y_rot = jit_pose_prep(states["x"], params["Cd"], k, states["h"])
        v_np = _numba_assemble_and_solve_tridiagonal_v(
            np.array(Y), np.array(mask, dtype=bool), np.array(Y_rot),
            np.array(states["s"]), np.array(params["sigmasq"]), sigmasq_loc,
            N, T, k, d
        )
        states["v"] = jnp.array(v_np)
        states["v"].block_until_ready()
        t_v = (time.perf_counter() - t0) * 1000.0

        # 6. Continuous Latents (x)
        t0 = time.perf_counter()
        states["x"] = fast_banded_gmrf_resample_x(
            jr.fold_in(seed, it * 20 + 5), Y=Y, mask=mask, v=states["v"], h=states["h"],
            s=states["s"], z=states["z"], Cd=params["Cd"], sigmasq=params["sigmasq"],
            Ab=params["Ab"], Q=params["Q"], nlags=nlags
        )
        states["x"].block_until_ready()
        t_x = (time.perf_counter() - t0) * 1000.0

        # 7. Scales (s)
        t0 = time.perf_counter()
        states["s"] = jit_resample_scales(
            jr.fold_in(seed, it * 20 + 6), Y=Y, x=states["x"], v=states["v"], h=states["h"],
            Cd=params["Cd"], sigmasq=params["sigmasq"], mask=mask,
            s_0=model.get("noise_prior", 1.0), nu_s=float(hypparams["obs_hypparams"]["nu_s"])
        )
        states["s"].block_until_ready()
        t_s = (time.perf_counter() - t0) * 1000.0

        total = (time.perf_counter() - t_tot) * 1000.0

        accounted = t_trans + t_ar + t_z + t_h + t_v + t_x + t_s
        print(f"Iter {it}: Total = {total:6.1f} ms | x: {t_x:5.1f}ms | v: {t_v:5.1f}ms | h: {t_h:5.1f}ms | z: {t_z:5.1f}ms | AR: {t_ar:5.1f}ms | s: {t_s:5.1f}ms | trans: {t_trans:5.1f}ms (overhead: {total-accounted:.1f}ms)")

if __name__ == "__main__":
    main()
