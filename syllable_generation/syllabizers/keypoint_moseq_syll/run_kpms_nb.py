import multiprocessing
import os
import pickle
import sys
from pathlib import Path

import numpy as np

# Disable CUDA for child workers
if multiprocessing.current_process().name != 'MainProcess':
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import logging
    logging.getLogger("jax._src.xla_bridge").setLevel(logging.CRITICAL)
    logging.getLogger("jax").setLevel(logging.CRITICAL)

# Monkeypatch numpy.bool8 for compatibility on newer numpy versions
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

import jax
import jax.numpy as jnp
import keypoint_moseq as kpms
from jax_moseq.models.keypoint_slds.alignment import preprocess_for_pca
from jax_moseq.models.keypoint_slds.initialize import estimate_error, init_states
from jax_moseq.utils import set_mixed_map_iters

# Set iterations
set_mixed_map_iters(64)

# ===== Memory Patches: Batched Resamplers =====
# Prevents RESOURCE_EXHAUSTED OOMs on large 64-bit datasets.

import jax.random as jr
from tqdm import tqdm

BATCH_SIZE = 100
SCALES_BS = min(BATCH_SIZE, 50)

# 1. Patch resample_discrete_stateseqs (Gibbs step for Z)
from jax_moseq.models.arhmm import gibbs as arhmm_gibbs

_orig_resample_discrete = arhmm_gibbs.resample_discrete_stateseqs

def _batched_resample_discrete_stateseqs(seed, x, mask, Ab, Q, pi, **kwargs):
    N = x.shape[0]
    results = []
    for i in tqdm(range(0, N, BATCH_SIZE), desc=f"resample_z (bs={BATCH_SIZE})", leave=False):
        end = min(i + BATCH_SIZE, N)
        batch_seed = jr.fold_in(seed, i)
        z_batch = _orig_resample_discrete(
            batch_seed, x[i:end], mask[i:end], Ab, Q, pi, **kwargs
        )
        z_batch.block_until_ready()
        results.append(z_batch)
    return jnp.concatenate(results, axis=0)

arhmm_gibbs.resample_discrete_stateseqs = _batched_resample_discrete_stateseqs
print(f"[Patch] batched resample_discrete_stateseqs active (batch_size={BATCH_SIZE})")


# 2. Patch resample_scales (Gibbs step for S - local noise scales)
from jax_moseq.models.keypoint_slds import gibbs as slds_gibbs
from jax_moseq.models.keypoint_slds import slds
from jax_moseq.models.keypoint_slds.gibbs import compute_squared_error


@jax.jit
def _resample_scales_kernel(seed, Y, x, v, h, Cd, sigmasq, nu_s, s_0):
    sqerr = compute_squared_error(Y, x, v, h, Cd)
    return slds.resample_scales_from_sqerr(seed, sqerr, sigmasq, nu_s, s_0)

def _batched_resample_scales(seed, Y, x, v, h, Cd, sigmasq, nu_s, s_0, **kwargs):
    N = Y.shape[0]
    results = []
    for i in tqdm(range(0, N, SCALES_BS), desc=f"resample_scales (bs={SCALES_BS})", leave=False):
        end = min(i + SCALES_BS, N)
        batch_seed = jr.fold_in(seed, i)
        s0_b = s_0[i:end] if (hasattr(s_0, 'shape') and len(s_0.shape) >= 3 and s_0.shape[0] == N) else s_0
        s_batch = _resample_scales_kernel(
            batch_seed, Y[i:end], x[i:end], v[i:end], h[i:end],
            Cd, sigmasq, nu_s, s0_b
        )
        s_batch.block_until_ready()
        results.append(s_batch)
    return jnp.concatenate(results, axis=0)

slds_gibbs.resample_scales = _batched_resample_scales
print(f"[Patch] batched resample_scales active (batch_size={SCALES_BS})")


# 3. Patch resample_continuous_stateseqs (Gibbs step for X - continuous latent states)
_orig_resample_continuous = slds_gibbs.resample_continuous_stateseqs

def _batched_resample_continuous_stateseqs(seed, Y, mask, v, h, s, z, Cd, sigmasq, Ab, Q, **kwargs):
    N = Y.shape[0]
    results = []
    for i in tqdm(range(0, N, BATCH_SIZE), desc=f"resample_x (bs={BATCH_SIZE})", leave=False):
        end = min(i + BATCH_SIZE, N)
        batch_seed = jr.fold_in(seed, i)
        x_part = _orig_resample_continuous(
            batch_seed,
            Y[i:end],
            mask[i:end],
            v[i:end],
            h[i:end],
            s[i:end],
            z[i:end],
            Cd,
            sigmasq,
            Ab,
            Q,
            **kwargs
        )
        x_part.block_until_ready()
        results.append(x_part)
    return jnp.concatenate(results, axis=0)

slds_gibbs.resample_continuous_stateseqs = _batched_resample_continuous_stateseqs
print(f"[Patch] batched resample_continuous_stateseqs active (batch_size={BATCH_SIZE})")
# ==========================================================

# Global configuration / settings
PROJECT_DIR = "karen_project_2"
DETECTOR = "deeplabcut"
SUBROUTINE = "jonathan"
CAM = "center_cam"          # or "side_cam"
DO_CALIBRATION = False      # Already have config with parameters
VIDEO_EXTENSION_OVERRIDE = None
P_THRESH = 0.9
FRAC_MIN_PARTS = 0.4
SMOOTH_WIN = 7
ENTER_T = 0.65
EXIT_T = 0.45
OFF_GRACE = 6
MIN_ON = 10
MIN_OFF = 6
DOWNSAMPLE = 2
NEW_FPS = 40

def make_masks(confidences):
    from tqdm import tqdm
    from utils.trimming import presence_mask_from_conf
    masks = {}
    for name, conf in tqdm(confidences.items(), desc="Generating presence masks"):
        masks[name] = presence_mask_from_conf(
            conf,
            p_thresh=P_THRESH,
            frac_min_parts=FRAC_MIN_PARTS,
            smooth_win=SMOOTH_WIN,
            enter_thresh=ENTER_T,
            exit_thresh=EXIT_T,
            off_grace_frames=OFF_GRACE,
            min_on_frames=MIN_ON,
            min_off_frames=MIN_OFF,
        )
    return masks

def batched_init_model_by_video(data, pca, config, params_subset_size=5, batch_size=20):
    N_videos = data['Y'].shape[0]
    print(f"Step 1: Initializing Params on first {params_subset_size} videos...")
    subset_slice = slice(0, params_subset_size)
    subset_data = {
        'Y': data['Y'][subset_slice],
        'mask': data['mask'][subset_slice],
        'conf': data['conf'][subset_slice] if 'conf' in data else None
    }
    model = kpms.init_model(subset_data, pca=pca, **config)

    print(f"Step 2: Initializing States for {N_videos} videos (Batch size: {batch_size})...")
    params = model['params']
    hypparams = model['hypparams']
    obs_hypparams = hypparams['obs_hypparams']
    seed = model['seed']
    error_estimator = config.get('error_estimator', None)

    anterior_idxs = config.get('anterior_idxs')
    posterior_idxs = config.get('posterior_idxs')
    conf_threshold = config.get('conf_threshold', 0.5)
    fix_heading = config.get('fix_heading', False)

    full_states = {'x': [], 'v': [], 'h': [], 's': [], 'z': []}
    num_batches = int(np.ceil(N_videos / batch_size))

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, N_videos)

        Y_chunk = jnp.array(data['Y'][start_idx:end_idx])
        mask_chunk = jnp.array(data['mask'][start_idx:end_idx])
        conf_chunk = jnp.array(data['conf'][start_idx:end_idx]) if 'conf' in data else None

        Y_flat_chunk, v_chunk, h_chunk = preprocess_for_pca(
            Y_chunk,
            anterior_idxs=anterior_idxs,
            posterior_idxs=posterior_idxs,
            conf=conf_chunk,
            conf_threshold=conf_threshold,
            fix_heading=fix_heading,
            verbose=False
        )

        if conf_chunk is not None and error_estimator is not None:
            batch_noise_prior = estimate_error(conf_chunk, **error_estimator)
        else:
            batch_noise_prior = 1.0

        chunk_states = init_states(
            seed,
            Y_chunk,
            mask_chunk,
            params,
            batch_noise_prior,
            obs_hypparams,
            Y_flat=Y_flat_chunk,
            v=v_chunk,
            h=h_chunk,
        )

        for k in ['x', 'v', 'h', 's', 'z']:
            if k in chunk_states:
                full_states[k].append(jax.device_put(chunk_states[k], jax.devices('cpu')[0]))

        print(f"  Batch {i+1}/{num_batches} (Videos {start_idx}-{end_idx}) done.")
        del Y_chunk, mask_chunk, conf_chunk, Y_flat_chunk, v_chunk, h_chunk, chunk_states, batch_noise_prior

    print("Step 3: Concatenating all states...")
    final_states = {}
    for k in ['x', 'v', 'h', 's', 'z']:
        if full_states[k]:
            final_states[k] = jnp.concatenate(full_states[k], axis=0)

    model['states'] = final_states

    if 'conf' in data and error_estimator is not None:
        print("Recalculating full noise_prior (on CPU)...")
        with jax.default_device(jax.devices("cpu")[0]):
             model['noise_prior'] = estimate_error(jnp.array(data['conf']), **error_estimator)
    else:
        model['noise_prior'] = 1.0

    print(f"Done. State 'x' shape: {final_states['x'].shape}")
    if 'z' in final_states:
        print(f"      State 'z' shape: {final_states['z'].shape}")
    return model

def inspect_structure(d, indent=0):
    spacer = "  " * indent
    if isinstance(d, dict):
        for k, v in d.items():
            if isinstance(v, dict):
                print(f"{spacer}['{k}']: <dict> with keys {list(v.keys())}")
                inspect_structure(v, indent + 1)
            elif hasattr(v, 'shape'):
                print(f"{spacer}['{k}']: shape={v.shape} | type={type(v).__name__}")
            else:
                val_str = str(v)
                if len(val_str) > 50: val_str = val_str[:50] + "..."
                print(f"{spacer}['{k}']: {type(v).__name__} = {val_str}")

def main():
    # 1. Environment tweaks
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

    # Repo discovery
    start = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd().resolve()
    for p in (start, *start.parents):
        if p.name == "syllable_generation":
            REPO_ROOT = p
            break
    else:
        REPO_ROOT = Path("/home/fsp585/Desktop/FiberphotometryCode/syllable_generation")

    PKG_PARENT = REPO_ROOT.parent
    if str(PKG_PARENT) not in sys.path:
        sys.path.insert(0, str(PKG_PARENT))

    # Load Pickle
    PKL_FILENAME = "/home/fsp585/Desktop/FiberphotometryCode/syllable_generation/data/sessions_karen_project_dirty.pkl"
    pkl_path = start / PKL_FILENAME
    print(f"--- Inspecting: {pkl_path} ---")
    if not pkl_path.exists():
        print(f"ERROR: File not found at {pkl_path}")
        sys.exit(1)

    with open(pkl_path, "rb") as f:
        sessions = pickle.load(f)

    print(f"Total Sessions Loaded: {len(sessions)}")

    from utils.paths import ref_pairs_from_sessions
    keypoint_paths, video_paths = ref_pairs_from_sessions(sessions, subroutine=SUBROUTINE, cam=CAM)
    VID_DIR = str(Path(video_paths[0]).parent)
    EXT = Path(video_paths[0]).suffix.lstrip(".").lower()

    coordinates, confidences, bodyparts = kpms.load_keypoints(keypoint_paths, DETECTOR)

    print(f"--- DOWNSAMPLING x{DOWNSAMPLE} ---")
    coordinates = {k: v[::DOWNSAMPLE] for k, v in coordinates.items()}
    confidences = {k: v[::DOWNSAMPLE] for k, v in confidences.items()}

    # Update config
    kpms.update_config(PROJECT_DIR, video_dir=VID_DIR, video_extension=EXT, fps=NEW_FPS)

    # Masks
    all_masks = make_masks(confidences)

    # Filter with masks
    from utils.trimming import clamp_conf_nonneg, filter_with_masks
    filtered_coords, filtered_confs, video_frame_indexes = filter_with_masks(
        coordinates, confidences, all_masks, PROJECT_DIR,
        return_video_frame_indexes=True, **kpms.load_config(PROJECT_DIR)
    )
    clamp_conf_nonneg(filtered_confs)

    # Filter sessions with < 5 valid frames to prevent AssertionError in format_data
    valid_keys = [k for k in filtered_coords.keys() if len(filtered_coords[k]) >= 5]
    removed_keys = [k for k in filtered_coords.keys() if len(filtered_coords[k]) < 5]
    if removed_keys:
        print(f"Removing {len(removed_keys)} sessions with less than 5 valid frames: {removed_keys}")
    filtered_coords = {k: filtered_coords[k] for k in valid_keys}
    filtered_confs = {k: filtered_confs[k] for k in valid_keys}
    video_frame_indexes = {k: video_frame_indexes[k] for k in valid_keys}

    for k in filtered_coords:
        assert len(filtered_coords[k]) == len(video_frame_indexes[k])
        assert len(filtered_confs[k])  == len(video_frame_indexes[k])

    # Fit PCA
    import gc
    try:
        pca = kpms.load_pca(PROJECT_DIR)
        print("Loaded existing PCA.")
    except Exception:
        print("Fitting new PCA (Memory Safe)...")
        all_Y = np.concatenate(list(filtered_coords.values()), axis=0)
        all_C = np.concatenate(list(filtered_confs.values()), axis=0)
        N = all_Y.shape[0]
        SAMPLE = 200000
        if N > SAMPLE:
            rng = np.random.default_rng(42)
            idx = np.sort(rng.choice(N, SAMPLE, replace=False))
            sub_Y = {"dummy": all_Y[idx]}
            sub_C = {"dummy": all_C[idx]}
        else:
            sub_Y = filtered_coords
            sub_C = filtered_confs

        sub_data, _ = kpms.format_data(sub_Y, sub_C, **kpms.load_config(PROJECT_DIR))
        pca = kpms.fit_pca(**sub_data, **kpms.load_config(PROJECT_DIR))
        kpms.save_pca(pca, PROJECT_DIR)
        del sub_data, sub_Y, sub_C, all_Y, all_C
        gc.collect()
        print("PCA Saved & Memory Cleared.")

    kpms.plot_scree(pca, project_dir=PROJECT_DIR)
    kpms.update_config(PROJECT_DIR, latent_dim=10)

    # Format data
    config_dict = kpms.load_config(PROJECT_DIR)
    data, metadata = kpms.format_data(filtered_coords, filtered_confs, **config_dict)

    # Update sigmasq_loc
    kpms.update_config(
        PROJECT_DIR,
        sigmasq_loc=kpms.estimate_sigmasq_loc(data["Y"], data["mask"], filter_size=config_dict["fps"])
    )

    # Initialize model
    model = batched_init_model_by_video(
        data,
        pca,
        kpms.load_config(PROJECT_DIR),
        params_subset_size=5,
        batch_size=20
    )

    print("--- Model Structure ---")
    inspect_structure(model)

    # Fit Model
    print("--- Fitting AR-only model (1 iteration) ---")
    model, model_name = kpms.fit_model(
        model, data, metadata, PROJECT_DIR,
        ar_only=True, num_iters=1, parallel_message_passing=False
    )

    print("--- Fitting Full model ---")
    model, data, metadata, current_iter = kpms.load_checkpoint(PROJECT_DIR, model_name, iteration=1)
    model = kpms.update_hypparams(model, kappa=1e4)
    model = kpms.fit_model(
        model, data, metadata, PROJECT_DIR, model_name,
        ar_only=False, start_iter=current_iter, num_iters=current_iter+1,
        parallel_message_passing=False
    )[0]
    print("--- FITTING COMPLETED SUCCESSFULLY ---")

if __name__ == '__main__':
    main()
