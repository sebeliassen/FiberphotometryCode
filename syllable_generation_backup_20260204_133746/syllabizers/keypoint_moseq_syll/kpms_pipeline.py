import os
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("DISPLAY", "")

import matplotlib
matplotlib.use("Agg", force=True)

from pathlib import Path
import numpy as np
import keypoint_moseq as kpms
from keypoint_moseq.viz import get_durations
import jax

# --- CUSTOM IMPORTS ---
from utils.paths import ref_pairs_from_sessions
from utils.trimming import presence_mask_from_conf, filter_with_masks, clamp_conf_nonneg, subset_after_filter
from utils.batch_init import batched_init_model_by_video
# Ensure this filename matches exactly what you have on disk
from utils.resample_batching import apply_optimizations 

def run_pipeline(
    sessions,
    project_dir,
    detector="deeplabcut",
    cam="center_cam",
    subroutine="jonathan",
    latent_dim=10,
    num_ar_iters=1,
    total_iters=1,
    override_video_extension=None,
    pca_mode="load",
    model_name=None,
    init_kappa=None,
    decrease_kappa_factor=None,
    subset_fraction=1.0,
    subset_blocks=5,
    subset_block_len=None,
    subset_seed=None,
    mixed_map_iters=24,
    seed=None,
    downsample_rate=1,  
    new_fps=None,
    use_batched_init=False,
    use_memory_patch=False
):
    # 0) Conditionally Apply the Memory Patch
    if use_memory_patch:
        print("[setup] Enabling memory optimization patch (Batched Resampling)...")
        apply_optimizations()
    else:
        print("[setup] Memory optimization patch disabled.")

    # 1) Derive file lists
    keypoint_paths, video_paths = ref_pairs_from_sessions(sessions, subroutine=subroutine, cam=cam)
    vid_dir = str(Path(video_paths[0]).parent)
    ext = (override_video_extension or Path(video_paths[0]).suffix.lstrip(".").lower())

    # 2) Load keypoints
    coordinates, confidences, bodyparts = kpms.load_keypoints(keypoint_paths, detector)

    # 3) Downsampling Config Update
    if downsample_rate > 1 and new_fps is not None:
        kpms.update_config(project_dir, fps=new_fps)

    # 4) Presence masks
    all_masks = {name: presence_mask_from_conf(conf) for name, conf in confidences.items()}

    # 5) Compress + outlier removal
    filtered_coords, filtered_confs, video_frame_indexes = filter_with_masks(
        coordinates, confidences, all_masks, project_dir,
        return_video_frame_indexes=True, downsample_rate=downsample_rate, 
        **kpms.load_config(project_dir)
    )
    clamp_conf_nonneg(filtered_confs)

    if subset_fraction < 1.0:
        filtered_coords, filtered_confs, video_frame_indexes = subset_after_filter(
            filtered_coords, filtered_confs, video_frame_indexes,
            frac=subset_fraction, n_blocks=subset_blocks,
            block_len=subset_block_len, seed=subset_seed,
        )

    # 6) Config
    kpms.update_config(project_dir, video_dir=vid_dir, video_extension=ext, latent_dim=latent_dim)

    # 7) Format data
    data, metadata = kpms.format_data(filtered_coords, filtered_confs, **kpms.load_config(project_dir))

    # 8) PCA
    if (subset_fraction is not None) and (subset_fraction < 1.0):
        pca = kpms.fit_pca(**data, **kpms.load_config(project_dir))
    elif pca_mode == "load":
        pca = kpms.load_pca(project_dir) 
    elif pca_mode == "load_or_fit":
        try:
            pca = kpms.load_pca(project_dir)
        except Exception:
            pca = kpms.fit_pca(**data, **kpms.load_config(project_dir))
            kpms.save_pca(pca, project_dir)
    else:
        raise ValueError(f"Unknown pca_mode={pca_mode!r}")
    
    # === Batching settings ===
    from jax_moseq.utils import set_mixed_map_iters
    import math
    eff_mixed_map_iters = mixed_map_iters
    if (subset_fraction is not None) and (subset_fraction < 1.0):
        eff_mixed_map_iters = max(1, math.ceil(mixed_map_iters * subset_fraction))
    set_mixed_map_iters(eff_mixed_map_iters)

    # === 9) Initialization & Checkpoint Loading ===
    requested_name = model_name
    model_name_used = requested_name 
    ckpt_path = Path(project_dir) / (requested_name or "NON_EXISTENT") / "checkpoint.h5"

    # [STEP 1] GET STATE (Load or Init)
    if requested_name and ckpt_path.exists():
        print(f"[resume] Checkpoint found. Loading latest state for '{requested_name}'...")
        # Load latest (do not specify iteration, let it find the max)
        model, data, metadata, current_iter = kpms.load_checkpoint(project_dir, requested_name)
        print(f"[resume] Loaded at iteration {current_iter}.")
    else:
        print(f"[init] Fresh start. Initializing model...")
        rng_key = jax.random.PRNGKey(int(seed)) if seed is not None else jax.random.PRNGKey(0)
        init_config = kpms.load_config(project_dir)

        if use_batched_init:
            print("[init] Using BATCHED initialization...")
            model = batched_init_model_by_video(
                data, pca, init_config, 
                params_subset_size=5, batch_size=20, seed=rng_key
            )
        else:
            print("[init] Using STANDARD initialization...")
            model = kpms.init_model(data, pca=pca, seed=rng_key, **init_config)
            
        if init_kappa is not None:
             model = kpms.update_hypparams(model, kappa=float(init_kappa))
             
        current_iter = 0

    # [STEP 2] PHASE 1: AR-ONLY FIT (Reach the AR Milestone)
    # We only run this if we haven't reached the AR iteration count yet.
    if current_iter < num_ar_iters:
        print(f"[phase 1] Current iter ({current_iter}) < AR target ({num_ar_iters}). Running AR fit...")
        
        model, model_name_used = kpms.fit_model(
            model, data, metadata, project_dir,
            model_name=requested_name,
            ar_only=True,
            start_iter=current_iter,  # Resume from wherever we are
            num_iters=num_ar_iters,   # Stop exactly at AR milestone
            parallel_message_passing=False,
        )
        current_iter = num_ar_iters
        print(f"[phase 1] AR fit complete. Reached iteration {current_iter}.")
    else:
        print(f"[phase 1] Skipping AR fit (Already at iter {current_iter} >= {num_ar_iters}).")

    # [STEP 3] TRANSITION (Update Kappa for SLDS)
    # Only update kappa if we are about to run SLDS
    slds_kappa = None
    if (init_kappa is not None) and (decrease_kappa_factor is not None):
        slds_kappa = float(init_kappa) / float(decrease_kappa_factor)
        print(f"[transition] Updating Kappa for SLDS: {slds_kappa}")
        model = kpms.update_hypparams(model, kappa=slds_kappa)

    # [STEP 4] PHASE 2: SLDS FIT (Reach the Final Milestone)
    # total_iters here means "How many SLDS steps to add"
    final_target_iter = num_ar_iters + total_iters
    
    if current_iter < final_target_iter:
        print(f"[phase 2] Starting SLDS fit from {current_iter} to {final_target_iter}...")
        
        # DIAGNOSTICS (Optional, keeping your request)
        print("\n--- DIAGNOSTICS: Pre-SLDS State ---")
        if 'noise_prior' in model:
             np_val = model['noise_prior']
             print(f"NOISE PRIOR: {np_val.shape if hasattr(np_val, 'shape') else np_val}")
        print("-----------------------------------\n")

        model = kpms.fit_model(
            model, data, metadata, project_dir, model_name_used,
            ar_only=False,
            start_iter=current_iter,
            num_iters=final_target_iter, # kpms expects the absolute end number
            parallel_message_passing=False,
        )[0]
        current_iter = final_target_iter
    else:
        print(f"[phase 2] Training already complete (Current {current_iter} >= Target {final_target_iter}).")

    # --- Summary metrics ---
    z = np.array(model["states"]["z"])
    mask = np.array(data["mask"])
    median_duration = float(np.median(get_durations(z, mask))) if mask.size else float("nan")
    frames_used = int(mask.sum())

    model_dir = os.path.join(project_dir, model_name_used)
    np.savez_compressed(
        os.path.join(model_dir, "video_frame_indexes.npz"),
        **{k: np.asarray(v, dtype=np.int64) for k, v in video_frame_indexes.items()}
    )

    return {
        "project_dir": project_dir,
        "video_dir": vid_dir,
        "frames_used": frames_used,
        "median_duration_frames": median_duration,
        "model_name": model_name_used,
        "iters_done": current_iter,
        "kappa_slds": slds_kappa,
    }