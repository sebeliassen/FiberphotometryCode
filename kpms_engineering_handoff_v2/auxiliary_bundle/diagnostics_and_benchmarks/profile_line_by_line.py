import os
import sys
import time
import numpy as np
np.bool8 = np.bool_
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

import keypoint_moseq as kpms
import jax_moseq.models.keypoint_slds.gibbs as slds_gibbs
import jax_moseq.models.arhmm.gibbs as arhmm_gibbs
from utils.paths import ref_pairs_from_sessions
from utils.trimming import presence_mask_from_conf, filter_with_masks, clamp_conf_nonneg
from utils.batch_init import batched_init_model_by_video
from utils.resample_batching import fast_banded_gmrf_resample_x, fast_resample_discrete_stateseqs, vectorized_resample_ar_params, _numba_assemble_and_solve_tridiagonal_v
from jax_moseq.utils.autoregression import get_nlags
from jax_moseq.utils.transitions import count_transitions, sample_hdp_transitions

SCRATCH_DIR = os.path.join(KPMS_DIR, "scratch_profiling_project")
DIRTY_PKL = os.path.join(REPO_ROOT, "syllable_generation/data/sessions_karen_project_dirty.pkl")

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
    print("      LINE-BY-LINE GIBBS ITERATION PROFILER (WHERE IS EVERY MILLISECOND?)")
    print("=" * 80)

    # Warmup
    _ = fast_resample_discrete_stateseqs(seed, states["x"], mask, params["Ab"], params["Q"], params["pi"], nlags=nlags, num_states=num_states)
    _ = fast_banded_gmrf_resample_x(seed, Y=Y, mask=mask, v=states["v"], h=states["h"], s=states["s"], z=states["z"], Cd=params["Cd"], sigmasq=params["sigmasq"], Ab=params["Ab"], Q=params["Q"], nlags=nlags)

    # Profile 3 consecutive iterations with microsecond precision
    for it in range(1, 4):
        print(f"\n--- Profiling Iteration {it} ---")
        t_tot = time.perf_counter()
        
        # 1. Transitions
        t0 = time.perf_counter()
        counts = count_transitions(num_states, states["z"], mask)
        counts.block_until_ready()
        t_count = (time.perf_counter() - t0) * 1000.0
        
        t0 = time.perf_counter()
        params["betas"], params["pi"] = sample_hdp_transitions(
            jr.fold_in(seed, it * 20 + 1), counts,
            params["betas"], hypparams["trans_hypparams"]["alpha"],
            hypparams["trans_hypparams"]["kappa"], hypparams["trans_hypparams"]["gamma"]
        )
        params["pi"].block_until_ready()
        t_hdp = (time.perf_counter() - t0) * 1000.0
        
        # 2. AR Dynamics
        t0 = time.perf_counter()
        params["Ab"], params["Q"] = arhmm_gibbs.resample_ar_params(
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
        states["h"] = slds_gibbs.resample_heading(
            jr.fold_in(seed, it * 20 + 4), Y, states["x"], states["v"], states["s"], params["Cd"], params["sigmasq"]
        )
        states["h"].block_until_ready()
        t_h = (time.perf_counter() - t0) * 1000.0

        # 5. Centroid Location (v)
        t0 = time.perf_counter()
        Y_aligned = slds_gibbs.estimate_aligned(states["x"], params["Cd"], k)
        Y_rot = slds_gibbs.apply_rotation(Y_aligned, states["h"])
        Y_rot.block_until_ready()
        t_rot = (time.perf_counter() - t0) * 1000.0
        
        t0 = time.perf_counter()
        v_np = _numba_assemble_and_solve_tridiagonal_v(
            np.array(Y), np.array(mask, dtype=bool), np.array(Y_rot),
            np.array(states["s"]), np.array(params["sigmasq"]), sigmasq_loc,
            N, T, k, d
        )
        states["v"] = jnp.array(v_np)
        states["v"].block_until_ready()
        t_v_solve = (time.perf_counter() - t0) * 1000.0
        
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
        states["s"] = slds_gibbs.resample_scales(
            jr.fold_in(seed, it * 20 + 6), Y=Y, x=states["x"], v=states["v"], h=states["h"],
            Cd=params["Cd"], sigmasq=params["sigmasq"], mask=mask,
            **hypparams["obs_hypparams"], s_0=model.get("noise_prior", 1.0)
        )
        states["s"].block_until_ready()
        t_s = (time.perf_counter() - t0) * 1000.0
        
        total = (time.perf_counter() - t_tot) * 1000.0
        
        print(f"  1. Transitions count + HDP:  {t_count + t_hdp:6.1f} ms  (count: {t_count:.1f}ms, hdp: {t_hdp:.1f}ms)")
        print(f"  2. AR Dynamics (Ab, Q):      {t_ar:6.1f} ms")
        print(f"  3. Discrete Syllables (z):   {t_z:6.1f} ms")
        print(f"  4. Heading (h):              {t_h:6.1f} ms")
        print(f"  5. Pose Rotation Prep:       {t_rot:6.1f} ms")
        print(f"  6. Centroid Tridiagonal Solve: {t_v_solve:6.1f} ms")
        print(f"  7. Continuous Latents (x GMRF): {t_x:6.1f} ms")
        print(f"  8. Noise Scales (s):         {t_s:6.1f} ms")
        accounted = t_count + t_hdp + t_ar + t_z + t_h + t_rot + t_v_solve + t_x + t_s
        print(f"  Sum of Sub-steps:            {accounted:6.1f} ms")
        print(f"  Total Measured Iteration:    {total:6.1f} ms  (Overhead / Framework: {total - accounted:.1f} ms)")

if __name__ == "__main__":
    main()
